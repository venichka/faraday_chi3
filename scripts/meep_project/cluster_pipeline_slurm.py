#!/usr/bin/env python3
"""Slurm-aware pipeline launcher for cavity optimization + sims + sweeps.

Usage pattern:
  1) Submit from login node (allocates and auto-releases resources):
     python cluster_pipeline_slurm.py --submit [runtime options...]

  2) Run inside an existing allocation:
     python cluster_pipeline_slurm.py [runtime options...]
"""

from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple


def now_tag() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def split_csv_tokens(text: str) -> List[str]:
    vals = []
    for part in str(text).split(","):
        token = part.strip().lower()
        if token:
            vals.append(token)
    return vals


def parse_dims(text: str) -> List[int]:
    dims = []
    for token in split_csv_tokens(text):
        dim = int(token)
        if dim not in (1, 3):
            raise ValueError(f"Invalid dim '{dim}'. Allowed: 1 or 3.")
        dims.append(dim)
    return sorted(set(dims))


def parse_optimizers(text: str) -> List[str]:
    vals = split_csv_tokens(text)
    allowed = {"new", "mf"}
    bad = [v for v in vals if v not in allowed]
    if bad:
        raise ValueError(f"Invalid optimizer(s): {bad}. Allowed: new,mf")
    if not vals:
        raise ValueError("At least one optimizer must be selected.")
    return vals


def parse_fidelity(text: str) -> List[str]:
    t = str(text).strip().lower()
    if t == "low":
        return ["low"]
    if t == "high":
        return ["high"]
    if t == "both":
        return ["low", "high"]
    raise ValueError("Fidelity must be one of: low, high, both.")


def shell_join(cmd: Sequence[str]) -> str:
    return shlex.join([str(x) for x in cmd])


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def stage_env() -> Dict[str, str]:
    env = os.environ.copy()
    env.setdefault("OMP_NUM_THREADS", "1")
    env.setdefault("OPENBLAS_NUM_THREADS", "1")
    env.setdefault("MKL_NUM_THREADS", "1")
    env.setdefault("NUMEXPR_NUM_THREADS", "1")
    env.setdefault("MPLCONFIGDIR", "/tmp/mplcache")
    return env


def run_stage(
    name: str,
    cmd: Sequence[str],
    cwd: Path,
    env: Dict[str, str],
    log_dir: Path,
    stage_times: Dict[str, Any],
) -> None:
    ensure_dir(log_dir)
    log_path = log_dir / f"{name}.log"
    t0 = time.time()
    with log_path.open("w", encoding="utf-8") as fh:
        fh.write(f"[stage] {name}\n")
        fh.write(f"[cmd] {shell_join(cmd)}\n\n")
        fh.flush()
        rc = subprocess.call(
            list(cmd),
            cwd=str(cwd),
            env=env,
            stdout=fh,
            stderr=subprocess.STDOUT,
        )
    dt = float(time.time() - t0)
    stage_times[name] = {
        "seconds": dt,
        "returncode": int(rc),
        "log": str(log_path),
        "cmd": list(map(str, cmd)),
    }
    if rc != 0:
        raise RuntimeError(f"Stage '{name}' failed (rc={rc}). See: {log_path}")


def run_parallel_stages(
    stages: Sequence[Tuple[str, Sequence[str]]],
    cwd: Path,
    env: Dict[str, str],
    log_dir: Path,
    stage_times: Dict[str, Any],
) -> None:
    ensure_dir(log_dir)
    procs = []
    t0s = {}
    for name, cmd in stages:
        log_path = log_dir / f"{name}.log"
        fh = log_path.open("w", encoding="utf-8")
        fh.write(f"[stage] {name}\n")
        fh.write(f"[cmd] {shell_join(cmd)}\n\n")
        fh.flush()
        proc = subprocess.Popen(
            list(cmd),
            cwd=str(cwd),
            env=env,
            stdout=fh,
            stderr=subprocess.STDOUT,
        )
        procs.append((name, cmd, proc, fh, log_path))
        t0s[name] = time.time()

    failed = []
    for name, cmd, proc, fh, log_path in procs:
        rc = proc.wait()
        fh.close()
        dt = float(time.time() - t0s[name])
        stage_times[name] = {
            "seconds": dt,
            "returncode": int(rc),
            "log": str(log_path),
            "cmd": list(map(str, cmd)),
        }
        if rc != 0:
            failed.append((name, rc, log_path))
    if failed:
        details = "; ".join([f"{n} rc={rc} ({p})" for n, rc, p in failed])
        raise RuntimeError(f"Parallel stages failed: {details}")


def material_args(ns: argparse.Namespace) -> List[str]:
    args = [
        "--materials",
        str(ns.materials),
        "--high-index-material",
        str(ns.high_index_material),
        "--nH",
        str(ns.nH),
        "--kH",
        str(ns.kH),
        "--high-index-n2",
        str(ns.high_index_n2),
        "--kappa-ref-lambda",
        str(ns.kappa_ref_lambda),
        "--nL",
        str(ns.nL),
    ]
    return args


def add_submission_flags(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--cluster-profile",
        choices=("custom", "debug", "release"),
        default="custom",
        help="Preset Slurm resource profile: debug=2 nodes/40 CPUs, release=5 nodes/100 CPUs.",
    )
    parser.add_argument("--submit", action="store_true", help="Submit to Slurm via sbatch.")
    parser.add_argument(
        "--run-in-allocation",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    parser.add_argument("--nodes", type=int, default=None, help="Slurm nodes for submission.")
    parser.add_argument("--ntasks-per-node", type=int, default=None)
    parser.add_argument("--cpus-per-task", type=int, default=None)
    parser.add_argument("--mem", type=str, default="0", help="Slurm memory request (e.g. 200G or 0).")
    parser.add_argument("--time-limit", type=str, default="24:00:00", help="Slurm walltime.")
    parser.add_argument("--partition", type=str, default="", help="Slurm partition.")
    parser.add_argument("--account", type=str, default="", help="Slurm account.")
    parser.add_argument("--qos", type=str, default="", help="Slurm QoS.")
    parser.add_argument("--constraint", type=str, default="", help="Slurm constraint.")
    parser.add_argument("--job-name", type=str, default="meep-pipeline")
    parser.add_argument("--slurm-output", type=str, default="slurm-%j.out")
    parser.add_argument("--slurm-error", type=str, default="")
    parser.add_argument(
        "--job-shell-init",
        type=str,
        default='eval "$(micromamba shell hook --shell bash)"',
        help="Shell init snippet run inside Slurm job before activation.",
    )
    parser.add_argument(
        "--job-env-activate",
        type=str,
        default="micromamba activate meep-mpi",
        help="Environment activation command run inside Slurm job.",
    )
    parser.add_argument(
        "--sbatch-extra",
        action="append",
        default=[],
        help="Additional raw sbatch option (repeatable), e.g. --sbatch-extra=--exclusive",
    )


def add_runtime_flags(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--project-root", type=str, default=str(Path(__file__).resolve().parent))
    parser.add_argument("--run-root", type=str, default="")
    parser.add_argument("--python-exe", type=str, default="python")
    parser.add_argument("--mpi-launcher", type=str, default="mpirun")

    parser.add_argument(
        "--preset",
        choices=("smoke", "full"),
        default="smoke",
        help="Smoke for quick sanity checks; full for high-fidelity defaults.",
    )
    parser.add_argument("--optimizers", type=str, default="new,mf", help="Comma list: new,mf")
    parser.add_argument("--optimizer-workers", type=int, default=6)
    parser.add_argument("--optimizer-debug", action="store_true")
    parser.add_argument("--objective-mode", choices=("quick", "full"), default="")
    parser.add_argument("--objective-resolution", type=int, default=0)
    parser.add_argument("--objective-decay-threshold", type=float, default=1e-4)
    parser.add_argument("--optimizer-pump-intensity", type=float, default=1e12)
    parser.add_argument("--probe-target-mode", choices=("exact", "band", "both"), default="both")
    parser.add_argument("--bayes-init", type=int, default=6)
    parser.add_argument("--bayes-iters", type=int, default=12)
    parser.add_argument("--bayes-batch-size", type=int, default=1)
    parser.add_argument("--bayes-candidates", type=int, default=256)
    parser.add_argument("--bayes-xi", type=float, default=0.01)
    parser.add_argument("--random-seed", type=int, default=0)
    parser.add_argument("--mf-probe-epsilon", type=float, default=0.02)
    parser.add_argument("--mf-stage1-per-n", type=int, default=6)
    parser.add_argument("--mf-stage2-topk", type=int, default=3)
    parser.add_argument("--mf-stage3-topk", type=int, default=2)
    parser.add_argument("--mf-disable-stage2", action="store_true")

    parser.add_argument("--skip-sims", action="store_true")
    parser.add_argument("--sim-dims", type=str, default="1,3", help="Comma list of dims.")
    parser.add_argument("--sim-fidelity", type=str, default="", help="low/high/both.")
    parser.add_argument("--sim-cutoff", type=float, default=1e-4)
    parser.add_argument("--sim-1d-res-low", type=int, default=40)
    parser.add_argument("--sim-1d-res-high", type=int, default=100)
    parser.add_argument("--sim-3d-res-low", type=int, default=20)
    parser.add_argument("--sim-3d-res-high", type=int, default=30)
    parser.add_argument("--sim-3d-mpi-ranks", type=int, default=16)

    parser.add_argument("--skip-sweeps", action="store_true")
    parser.add_argument("--sweep-dims", type=str, default="1,3", help="Comma list of dims.")
    parser.add_argument("--sweep-fidelity", type=str, default="", help="low/high/both.")
    parser.add_argument("--sweep-cutoff", type=float, default=1e-4)
    parser.add_argument("--sweep-range-scale", choices=("log", "linear"), default="log")
    parser.add_argument("--sweep-i-min", type=float, default=1e8)
    parser.add_argument("--sweep-i-max", type=float, default=2e12)
    parser.add_argument("--sweep-points", type=int, default=6)
    parser.add_argument("--sweep-1d-workers", type=int, default=6)
    parser.add_argument("--sweep-3d-workers", type=int, default=1)
    parser.add_argument("--sweep-1d-res-low", type=int, default=40)
    parser.add_argument("--sweep-1d-res-high", type=int, default=100)
    parser.add_argument("--sweep-3d-res-low", type=int, default=20)
    parser.add_argument("--sweep-3d-res-high", type=int, default=30)
    parser.add_argument("--sweep-3d-mpi-ranks", type=int, default=16)
    parser.add_argument(
        "--parallel-sweep-dims",
        action="store_true",
        help="Run 1D and 3D sweep commands in parallel (resource heavy).",
    )

    parser.add_argument("--seed-geometry", type=str, default="optimized_geometry_bayes_w6_signal_new.json")

    parser.add_argument("--materials", choices=("library", "constant", "fit"), default="constant")
    parser.add_argument("--high-index-material", type=str, default="tio2")
    parser.add_argument("--nH", type=float, default=2.31)
    parser.add_argument("--kH", type=float, default=8e-6)
    parser.add_argument("--nL", type=float, default=1.45)
    parser.add_argument("--high-index-n2", type=float, default=2.3e-18)
    parser.add_argument("--kappa-ref-lambda", type=float, default=1.55)


def apply_preset(ns: argparse.Namespace) -> None:
    if ns.preset == "smoke":
        if not ns.objective_mode:
            ns.objective_mode = "quick"
        if ns.objective_resolution <= 0:
            ns.objective_resolution = 40
        if not ns.sim_fidelity:
            ns.sim_fidelity = "low"
        if not ns.sweep_fidelity:
            ns.sweep_fidelity = "low"
        ns.bayes_init = min(ns.bayes_init, 4)
        ns.bayes_iters = min(ns.bayes_iters, 4)
    else:
        if not ns.objective_mode:
            ns.objective_mode = "full"
        if ns.objective_resolution <= 0:
            ns.objective_resolution = 60
        if not ns.sim_fidelity:
            ns.sim_fidelity = "high"
        if not ns.sweep_fidelity:
            ns.sweep_fidelity = "high"


def apply_cluster_profile(ns: argparse.Namespace) -> None:
    if ns.cluster_profile == "debug":
        if ns.nodes is None:
            ns.nodes = 2
        if ns.ntasks_per_node is None:
            ns.ntasks_per_node = 20
        if ns.cpus_per_task is None:
            ns.cpus_per_task = 1
        return

    if ns.cluster_profile == "release":
        if ns.nodes is None:
            ns.nodes = 5
        if ns.ntasks_per_node is None:
            ns.ntasks_per_node = 20
        if ns.cpus_per_task is None:
            ns.cpus_per_task = 1
        return

    if ns.nodes is None:
        ns.nodes = 1
    if ns.ntasks_per_node is None:
        ns.ntasks_per_node = 1
    if ns.cpus_per_task is None:
        ns.cpus_per_task = 16


def bool_in_slurm() -> bool:
    return bool(os.environ.get("SLURM_JOB_ID"))


def submit_to_slurm(ns: argparse.Namespace, remaining_run_args: List[str]) -> None:
    script_path = Path(__file__).resolve()
    sbatch_cmd = [
        "sbatch",
        "--job-name",
        ns.job_name,
        "--nodes",
        str(ns.nodes),
        "--ntasks-per-node",
        str(ns.ntasks_per_node),
        "--cpus-per-task",
        str(ns.cpus_per_task),
        "--time",
        ns.time_limit,
        "--mem",
        ns.mem,
        "--output",
        ns.slurm_output,
    ]
    if ns.slurm_error:
        sbatch_cmd += ["--error", ns.slurm_error]
    if ns.partition:
        sbatch_cmd += ["--partition", ns.partition]
    if ns.account:
        sbatch_cmd += ["--account", ns.account]
    if ns.qos:
        sbatch_cmd += ["--qos", ns.qos]
    if ns.constraint:
        sbatch_cmd += ["--constraint", ns.constraint]
    for extra in ns.sbatch_extra:
        if extra:
            sbatch_cmd.append(str(extra))

    run_args = ["--run-in-allocation", *remaining_run_args]
    py = str(Path(ns.python_exe).expanduser())
    inner_cmd = [py, str(script_path), *run_args]
    wrap_steps: List[str] = []
    if str(ns.job_shell_init).strip():
        wrap_steps.append(str(ns.job_shell_init).strip())
    if str(ns.job_env_activate).strip():
        wrap_steps.append(str(ns.job_env_activate).strip())
    wrap_steps.append(f"cd {shlex.quote(str(Path(ns.project_root).resolve()))}")
    wrap_steps.append(shell_join(inner_cmd))
    wrap_payload = " && ".join(wrap_steps)
    sbatch_cmd += ["--wrap", f"bash -lc {shlex.quote(wrap_payload)}"]

    print("[submit]", shell_join(sbatch_cmd), flush=True)
    proc = subprocess.run(sbatch_cmd, text=True)
    if proc.returncode != 0:
        raise SystemExit(proc.returncode)


def build_optimizer_cmds(
    ns: argparse.Namespace,
    py: str,
    run_root: Path,
    optimizer_names: List[str],
) -> Dict[str, Sequence[str]]:
    mats = material_args(ns)
    cmds: Dict[str, Sequence[str]] = {}
    for name in optimizer_names:
        out_dir = run_root / "optimizers" / name
        ensure_dir(out_dir)
        base_args = [
            "--optimizer",
            "bayes",
            "--workers",
            str(int(ns.optimizer_workers)),
            "--objective-mode",
            str(ns.objective_mode),
            "--objective-resolution",
            str(int(ns.objective_resolution)),
            "--objective-decay-threshold",
            str(float(ns.objective_decay_threshold)),
            "--pump-intensity",
            str(float(ns.optimizer_pump_intensity)),
            "--probe-target-mode",
            str(ns.probe_target_mode),
            "--bayes-init",
            str(int(ns.bayes_init)),
            "--bayes-iters",
            str(int(ns.bayes_iters)),
            "--bayes-batch-size",
            str(int(ns.bayes_batch_size)),
            "--bayes-candidates",
            str(int(ns.bayes_candidates)),
            "--bayes-xi",
            str(float(ns.bayes_xi)),
            "--random-seed",
            str(int(ns.random_seed)),
            "--out-geom",
            str(out_dir / "optimized_geometry.json"),
            "--out-modes",
            str(out_dir / "cavity_modes.json"),
            "--out-report",
            str(out_dir / "optimize_report.json"),
            "--eval-root",
            str(out_dir / ".opt_eval_tmp"),
            *mats,
        ]
        if ns.optimizer_debug:
            base_args += ["--debug", "--debug-prefix", str(out_dir / "optimize_debug")]

        if name == "new":
            cmds[name] = [py, "optimize_cavity_geometry.py", *base_args]
        elif name == "mf":
            mf_args = [
                "--probe-epsilon",
                str(float(ns.mf_probe_epsilon)),
                "--mf-stage1-per-n",
                str(int(ns.mf_stage1_per_n)),
                "--mf-stage2-topk",
                str(int(ns.mf_stage2_topk)),
                "--mf-stage3-topk",
                str(int(ns.mf_stage3_topk)),
            ]
            if ns.mf_disable_stage2:
                mf_args.append("--mf-disable-stage2")
            cmds[name] = [py, "optimize_cavity_geometry_mf.py", *mf_args, *base_args]
    return cmds


def sim_mode_for_fidelity(fidelity: str) -> str:
    return "quick" if fidelity == "low" else "full"


def sim_res(ns: argparse.Namespace, dim: int, fidelity: str) -> int:
    if dim == 1:
        return int(ns.sim_1d_res_low if fidelity == "low" else ns.sim_1d_res_high)
    return int(ns.sim_3d_res_low if fidelity == "low" else ns.sim_3d_res_high)


def sweep_res(ns: argparse.Namespace, dim: int, fidelity: str) -> int:
    if dim == 1:
        return int(ns.sweep_1d_res_low if fidelity == "low" else ns.sweep_1d_res_high)
    return int(ns.sweep_3d_res_low if fidelity == "low" else ns.sweep_3d_res_high)


def maybe_wrap_mpi(cmd: Sequence[str], dim: int, ranks: int, launcher: str) -> Sequence[str]:
    if int(dim) != 3:
        return list(cmd)
    return [launcher, "-np", str(int(ranks)), *cmd]


def build_report(
    run_root: Path,
    stage_times: Dict[str, Any],
    optimizer_names: List[str],
    sim_dims: List[int],
    sim_fids: List[str],
    sweep_dims: List[int],
    sweep_fids: List[str],
) -> Path:
    report = run_root / "cluster_pipeline_report.md"
    lines: List[str] = []
    lines.append("# Cluster Pipeline Report")
    lines.append("")
    lines.append(f"- Run root: `{run_root}`")
    lines.append(f"- Generated: `{datetime.utcnow().isoformat()}Z`")
    lines.append("")
    lines.append("## Stage Timings")
    lines.append("")
    lines.append("| Stage | rc | sec | log |")
    lines.append("|---|---:|---:|---|")
    for key in sorted(stage_times.keys()):
        rec = stage_times[key]
        lines.append(
            f"| {key} | {rec.get('returncode', 'n/a')} | "
            f"{float(rec.get('seconds', float('nan'))):.3f} | `{rec.get('log', '')}` |"
        )
    lines.append("")
    lines.append("## Expected Artifacts")
    lines.append("")
    for opt in optimizer_names:
        lines.append(f"### Optimizer `{opt}`")
        lines.append(f"- `{(run_root / 'optimizers' / opt / 'optimized_geometry.json')}`")
        lines.append(f"- `{(run_root / 'optimizers' / opt / 'cavity_modes.json')}`")
        lines.append(f"- `{(run_root / 'optimizers' / opt / 'optimize_report.json')}`")
    lines.append("")
    for opt in optimizer_names:
        lines.append(f"### Simulations `{opt}`")
        for fid in sim_fids:
            for dim in sim_dims:
                lines.append(f"- `{(run_root / 'sims' / opt / f'dim{dim}_{fid}' / 'faraday_summary.json')}`")
    lines.append("")
    for opt in optimizer_names:
        lines.append(f"### Sweeps `{opt}`")
        for fid in sweep_fids:
            for dim in sweep_dims:
                lines.append(f"- `{(run_root / 'sweeps' / opt / fid / 'pump_intensity_sweep_report.json')}`")
    report.write_text("\n".join(lines), encoding="utf-8")
    return report


def run_pipeline(ns: argparse.Namespace) -> None:
    apply_preset(ns)
    optimizer_names = parse_optimizers(ns.optimizers)
    sim_dims = parse_dims(ns.sim_dims)
    sim_fids = parse_fidelity(ns.sim_fidelity)
    sweep_dims = parse_dims(ns.sweep_dims)
    sweep_fids = parse_fidelity(ns.sweep_fidelity)

    project_root = Path(ns.project_root).resolve()
    run_root = Path(ns.run_root).resolve() if ns.run_root else (project_root / f"pipeline_cluster_{now_tag()}").resolve()
    ensure_dir(run_root)
    ensure_dir(run_root / "logs")

    py = str(Path(ns.python_exe).expanduser())
    env = stage_env()
    stage_times: Dict[str, Any] = {}

    optimizer_cmds = build_optimizer_cmds(ns, py, run_root, optimizer_names)
    for name in optimizer_names:
        run_stage(
            f"opt_{name}",
            optimizer_cmds[name],
            cwd=project_root,
            env=env,
            log_dir=run_root / "logs",
            stage_times=stage_times,
        )

    mats = material_args(ns)

    if not ns.skip_sims:
        for opt in optimizer_names:
            geom = run_root / "optimizers" / opt / "optimized_geometry.json"
            modes = run_root / "optimizers" / opt / "cavity_modes.json"
            for fid in sim_fids:
                mode = sim_mode_for_fidelity(fid)
                for dim in sim_dims:
                    res = sim_res(ns, dim, fid)
                    out_dir = ensure_dir(run_root / "sims" / opt / f"dim{dim}_{fid}")
                    base_cmd = [
                        py,
                        "faraday_meep_fp_circ.py",
                        "--mode",
                        mode,
                        "--dim",
                        str(dim),
                        "--resolution",
                        str(res),
                        "--decay-threshold",
                        str(float(ns.sim_cutoff)),
                        "--pump-intensity",
                        str(float(ns.optimizer_pump_intensity)),
                        "--geometry-file",
                        str(geom),
                        "--cavity-modes-file",
                        str(modes),
                        "--output-dir",
                        str(out_dir),
                        *mats,
                    ]
                    cmd = maybe_wrap_mpi(base_cmd, dim=dim, ranks=ns.sim_3d_mpi_ranks, launcher=ns.mpi_launcher)
                    run_stage(
                        f"sim_{opt}_dim{dim}_{fid}",
                        cmd,
                        cwd=project_root,
                        env=env,
                        log_dir=run_root / "logs",
                        stage_times=stage_times,
                    )

    if not ns.skip_sweeps:
        for opt in optimizer_names:
            geom = run_root / "optimizers" / opt / "optimized_geometry.json"
            modes = run_root / "optimizers" / opt / "cavity_modes.json"
            for fid in sweep_fids:
                mode = sim_mode_for_fidelity(fid)
                stage_batch: List[Tuple[str, Sequence[str]]] = []
                for dim in sweep_dims:
                    res = sweep_res(ns, dim, fid)
                    workers = int(ns.sweep_1d_workers if dim == 1 else ns.sweep_3d_workers)
                    out_root = ensure_dir(run_root / "sweeps" / opt / fid / f"dim{dim}")
                    base_cmd = [
                        py,
                        "pump_intensity_sweep.py",
                        "--dim",
                        str(dim),
                        "--intensity-range",
                        str(float(ns.sweep_i_min)),
                        str(float(ns.sweep_i_max)),
                        str(int(ns.sweep_points)),
                        "--range-scale",
                        str(ns.sweep_range_scale),
                        "--workers",
                        str(workers),
                        "--mode",
                        mode,
                        "--resolution",
                        str(res),
                        "--decay-threshold",
                        str(float(ns.sweep_cutoff)),
                        "--geometry-file",
                        str(geom),
                        "--cavity-modes-file",
                        str(modes),
                        "--output-root",
                        str(out_root),
                        *mats,
                    ]
                    cmd = maybe_wrap_mpi(
                        base_cmd,
                        dim=dim,
                        ranks=ns.sweep_3d_mpi_ranks,
                        launcher=ns.mpi_launcher,
                    )
                    stage_name = f"sweep_{opt}_dim{dim}_{fid}"
                    stage_batch.append((stage_name, cmd))

                if ns.parallel_sweep_dims and len(stage_batch) > 1:
                    run_parallel_stages(
                        stage_batch,
                        cwd=project_root,
                        env=env,
                        log_dir=run_root / "logs",
                        stage_times=stage_times,
                    )
                else:
                    for stage_name, cmd in stage_batch:
                        run_stage(
                            stage_name,
                            cmd,
                            cwd=project_root,
                            env=env,
                            log_dir=run_root / "logs",
                            stage_times=stage_times,
                        )

    report = build_report(
        run_root=run_root,
        stage_times=stage_times,
        optimizer_names=optimizer_names,
        sim_dims=sim_dims,
        sim_fids=sim_fids,
        sweep_dims=sweep_dims,
        sweep_fids=sweep_fids,
    )
    (run_root / "cluster_pipeline_stage_times.json").write_text(
        json.dumps(stage_times, indent=2), encoding="utf-8"
    )
    print(f"[run-root] {run_root}")
    print(f"[report] {report}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Cluster-ready full pipeline launcher.")
    add_submission_flags(parser)
    add_runtime_flags(parser)
    ns = parser.parse_args()
    apply_cluster_profile(ns)

    # Build runtime argument list for sbatch submission by filtering out submit-only flags.
    submit_only = {
        "--cluster-profile",
        "--submit",
        "--nodes",
        "--ntasks-per-node",
        "--cpus-per-task",
        "--mem",
        "--time-limit",
        "--partition",
        "--account",
        "--qos",
        "--constraint",
        "--job-name",
        "--slurm-output",
        "--slurm-error",
        "--job-shell-init",
        "--job-env-activate",
        "--sbatch-extra",
    }
    raw = sys.argv[1:]
    runtime_args: List[str] = []
    i = 0
    while i < len(raw):
        tok = raw[i]
        key = tok.split("=", 1)[0]
        if key in submit_only:
            if "=" in tok:
                i += 1
                continue
            # consume value for key-value options
            if key in {
                "--cluster-profile",
                "--nodes",
                "--ntasks-per-node",
                "--cpus-per-task",
                "--mem",
                "--time-limit",
                "--partition",
                "--account",
                "--qos",
                "--constraint",
                "--job-name",
                "--slurm-output",
                "--slurm-error",
                "--job-shell-init",
                "--job-env-activate",
                "--sbatch-extra",
            }:
                i += 2
            else:
                i += 1
            continue
        runtime_args.append(tok)
        i += 1

    if ns.submit and (not ns.run_in_allocation):
        submit_to_slurm(ns, runtime_args)
        return

    try:
        run_pipeline(ns)
    except Exception as exc:
        print(f"[error] {exc}", file=sys.stderr)
        raise


if __name__ == "__main__":
    main()
