#!/usr/bin/env python
"""Stage 7 -- review of the fringe model: which physics and instrument effects does the data need?

Stage 3's model (one Gaussian envelope shared by the effect and a fringe at the delayed pump's
1620 nm carrier) misfits the delay scans badly. This stage builds the model up one ingredient
at a time, fits BOTH delay scans jointly, and scores every variant on the same footing.

Model for scan k (u = tau - tau0_k, all envelopes peak-normalised):

  m_k(tau) = P_k + E_k He(u) + Hf(u) [c_k cos phi(u) + s_k sin phi(u)]   (+ a 2nd fringe if asked)
  phi(u)   = 2 pi f u + beta u^2                                          (carrier + linear chirp)
  H(u)     = G(u) + r [G(u - T) + G(u + T)]                                (cavity-echo satellites)
  G        = Gaussian of FWHM w, optionally exponentially modified (tail t_tail: ring-down)
  lock-in  y_n = (1 - a) m_n + a y_{n-1}  in acquisition order (a = 0: settled between points)

Linear in (P_k, E_k, c_k, s_k): solved exactly for every trial of the nonlinear parameters
(variable projection), which are optimised by multi-start least squares. Shared across scans:
everything physical (f, beta, envelope shapes, echo, lock-in a); per scan: tau0_k (timing drift)
and the linear amplitudes (the fringe phase is not stable between scans).

Noise: sigma_k from outside the overlap (stage 1). An optional random-phase fringe component
adds variance (1/2) A_r^2 Hf(u)^2 -- the "partial coherence" variant -- scored by -2 ln L.

Scores: chi2 (or -2 ln L), AIC = -2lnL + 2p, BIC = -2lnL + p ln n, and CROSS-SCAN validation:
the shared physics fitted on one scan predicts the other with only its own tau0 and linear
amplitudes refitted -- a model that merely overfits one scan fails this.

  python s7_models.py      -> figs/combined/s7_models_*.png, results/combined/s7_result.json
"""
import json
import time

import numpy as np
from scipy.optimize import least_squares
from scipy.special import erfcx

import common as K

TAGS = K.DELAY_TAGS
FWHM2SIG = 1.0 / (2.0 * np.sqrt(2.0 * np.log(2.0)))


# ---------------------------------------------------------------------------- building blocks
def gauss_env(u, w, tail):
    """Peak-normalised Gaussian (FWHM w), exponentially modified by a one-sided tail of time
    constant |tail| (positive: tail towards +u; 0: plain Gaussian)."""
    s = w * FWHM2SIG
    if abs(tail) < 1e-3:
        return np.exp(-0.5 * (u / s) ** 2)
    x = u if tail > 0 else -u
    lam = abs(tail)
    z = (s / lam - x / s) / np.sqrt(2.0)
    # EMG = 0.5 exp(-x^2/2s^2) erfcx(z), in log space: erfcx(z) ~ 2 exp(z^2) for z << 0
    with np.errstate(over="ignore", invalid="ignore"):
        lerf = np.where(z > -25.0, np.log(erfcx(np.maximum(z, -25.0))), np.log(2.0) + z ** 2)
    lg = -0.5 * (x / s) ** 2 + lerf
    return np.exp(lg - np.max(lg))


def envelope(u, w, tail, r, T):
    g = gauss_env(u, w, tail)
    if r != 0.0 and T > 0:
        g = g + r * (gauss_env(u - T, w, tail) + gauss_env(u + T, w, tail))
    return g


def lockin(m, a):
    """First-order lock-in lag across points, acquisition order = increasing delay."""
    if a <= 0:
        return m
    out = np.empty_like(m)
    out[0] = m[0]
    for i in range(1, len(m)):
        out[i] = (1 - a) * m[i] + a * out[i - 1]
    return out


# ---------------------------------------------------------------------------- the model family
class Model:
    """A variant = which nonlinear ingredients are free. `spec` keys:
      f0        starting carrier frequency (1/fs) -- or a list of starts
      free      names of shared nonlinear params that are fitted: subset of
                {f, beta, w_f, dtau_f, tail, r, T, a, f2}
      sep_env   effect has its own width w_e (else w_e = w_f)
      fringe2   add a second fringe at frequency f2
      random    add a random-phase fringe variance (partial coherence)
    """

    def __init__(self, name, spec):
        self.name, self.spec = name, spec

    # parameter vector layout -------------------------------------------------------------
    def layout(self, n_scans):
        names = ["tau0_%d" % k for k in range(n_scans)] + ["w_f"]
        if self.spec.get("sep_env"):
            names += ["w_e", "dtau_e"]
        for p in ("f", "beta", "tail", "r", "T", "a", "f2", "Ar"):
            if p in self.spec["free"]:
                names.append(p)
        return names

    def unpack(self, x, n_scans):
        d = dict(zip(self.layout(n_scans), x))
        d.setdefault("f", self.spec.get("f_fixed", 0.0))
        for p, v in (("beta", 0.0), ("tail", 0.0), ("r", 0.0), ("T", 0.0), ("a", 0.0), ("Ar", 0.0)):
            d.setdefault(p, v)
        d.setdefault("w_e", d["w_f"])
        d.setdefault("dtau_e", 0.0)
        return d

    def n_linear(self):
        return 2 + 2 * len(self.spec.get("harmonics", (1,))) + (2 if self.spec.get("fringe2") else 0)

    # design matrix for one scan ------------------------------------------------------------
    def design(self, t, tau0, p):
        u = t - tau0
        hf = envelope(u, p["w_f"], p["tail"], p["r"], p["T"])
        he = envelope(u - p["dtau_e"], p["w_e"], p["tail"], p["r"], p["T"])
        cols = [np.ones_like(t), he]
        for n in self.spec.get("harmonics", (1,)):
            ph = 2 * np.pi * n * p["f"] * u + n * p["beta"] * u ** 2
            cols += [hf * np.cos(ph), hf * np.sin(ph)]
        if self.spec.get("fringe2"):
            ph2 = 2 * np.pi * p["f2"] * u
            cols += [hf * np.cos(ph2), hf * np.sin(ph2)]
        A = np.column_stack([lockin(c, p["a"]) for c in cols])
        return A, hf

    def scan_fit(self, sc, tau0, p):
        """Weighted LS for the linear params of one scan; returns (resid/sd, coef, var)."""
        A, hf = self.design(sc["t"], tau0, p)
        var = sc["sig"] ** 2 + 0.5 * (p["Ar"] * hf) ** 2
        wt = 1.0 / np.sqrt(var)
        coef, *_ = np.linalg.lstsq(A * wt[:, None], sc["y"] * wt, rcond=None)
        return (sc["y"] - A @ coef) * wt, coef, var

    def residuals(self, x, scans):
        p = self.unpack(x, len(scans))
        out = []
        for k, sc in enumerate(scans):
            r, _, var = self.scan_fit(sc, p["tau0_%d" % k], p)
            out.append(r)
            if self.spec.get("random"):
                # -2lnL = sum r^2 + sum ln var; express ln var as an extra residual per point
                out.append(np.sqrt(np.maximum(np.log(var / sc["sig"] ** 2), 0.0)))
        return np.concatenate(out)


class TemplateModel(Model):
    """The SIMULATED fringe of the as-built sample (sim_labpoint.py, 1D, 125 fs pulses) used as
    the fringe shape: its envelope |C1(v)|, its carrier 1/T2 (1620 nm) and its phase drift psi0(v)
    (the effective carrier and chirp the cavity imposes) are all fixed by the simulation.
    Free: tau0_k, a complex fringe scale per scan (amplitude + absolute phase), the effect on its
    own Gaussian envelope, and a delay-axis scale kappa: lab delay u maps to sim delay
    v = sign * kappa * u. kappa = 1 means the lab's delay axis is right and the physics is as
    simulated; the sign covers the unknown delay direction."""

    TPL = None

    @classmethod
    def load_template(cls):
        if cls.TPL is None:
            import sim_labpoint as SL
            T2 = SL.C.carrier_period_fs(SL.F_DELAYED)
            rows = [SL.collect("trace", "best", SL.TRACE_FWHM, float(t)) for t in SL.TRACE_TAUS]
            rows = [r for r in rows if r]
            v = np.array([r["tau_fs"] for r in rows])
            c1 = np.array([r["fringe_re"] + 1j * r["fringe_im"] for r in rows])
            psi0 = np.unwrap(np.angle(c1 * np.exp(-2j * np.pi * v / T2)))
            cls.TPL = {"v": v, "amp": np.abs(c1) / np.abs(c1).max(), "psi0": psi0, "T2": T2,
                       "amp_deg_1d": float(np.abs(c1).max()),
                       "effect_1d": np.array([r["effect_deg"] for r in rows])}
        return cls.TPL

    def layout(self, n_scans):
        names = ["tau0_%d" % k for k in range(n_scans)] + ["w_e", "dtau_e"]
        if "kappa" in self.spec["free"]:
            names.append("kappa")
        for p in ("a", "Ar"):
            if p in self.spec["free"]:
                names.append(p)
        return names

    def unpack(self, x, n_scans):
        d = dict(zip(self.layout(n_scans), x))
        d.setdefault("kappa", 1.0)
        for p, v in (("a", 0.0), ("Ar", 0.0), ("beta", 0.0), ("tail", 0.0), ("r", 0.0), ("T", 0.0)):
            d.setdefault(p, v)
        d["w_f"] = d.get("w_f", 0.0)
        d["f"] = d["kappa"] / self.load_template()["T2"]
        return d

    def design(self, t, tau0, p):
        tp = self.load_template()
        u = t - tau0
        v = self.spec.get("sign", 1.0) * p["kappa"] * u
        amp = np.interp(v, tp["v"], tp["amp"], left=0.0, right=0.0)
        psi = np.interp(v, tp["v"], tp["psi0"])
        ph = 2 * np.pi * v / tp["T2"] + psi
        he = gauss_env(u - p["dtau_e"], p["w_e"], 0.0)
        cols = [np.ones_like(t), he, amp * np.cos(ph), amp * np.sin(ph)]
        A = np.column_stack([lockin(c, p["a"]) for c in cols])
        return A, amp


class TiedModel:
    """Physically constrained version of a model: ONE effect amplitude and ONE amplitude per fringe
    component shared by both scans (same sample, same powers, a minute apart); only the pedestal,
    tau0 and each fringe's absolute phase may differ between scans. Warm-started from the untied
    fit. Wraps an untied `base` model for the envelopes and carriers."""

    def __init__(self, base):
        self.base = base
        self.name = base.name.split()[0] + "t " + " ".join(base.name.split()[1:]) + " [tied]"
        self.spec = base.spec

    def n_comp(self):
        return (self.base.n_linear() - 2) // 2

    def layout(self, n_scans):
        return self.base.layout(n_scans) + ["phi%d_%d" % (j, k) for j in range(self.n_comp())
                                            for k in range(n_scans)]

    def unpack(self, x, n_scans):
        nb = len(self.base.layout(n_scans))
        d = self.base.unpack(x[:nb], n_scans)
        for name, v in zip(self.layout(n_scans)[nb:], x[nb:]):
            d[name] = v
        return d

    def n_linear(self):
        return None                      # scored separately (joint linear system)

    def joint(self, x, scans, fixed=None):
        """Joint weighted LS over scans. `fixed` = (E, [A_j]) holds the shared amplitudes (for
        cross-validation); returns (resid list, coef, var list, per-scan columns)."""
        p = self.unpack(x, len(scans))
        nS, nc = len(scans), self.n_comp()
        rows, ys, ws, varl, colsl = [], [], [], [], []
        for k, sc in enumerate(scans):
            A, hf = self.base.design(sc["t"], p["tau0_%d" % k], p)
            var = sc["sig"] ** 2 + 0.5 * (p["Ar"] * hf) ** 2
            cols = np.zeros((len(sc["t"]), nS + 1 + nc))
            cols[:, k] = A[:, 0]
            cols[:, nS] = A[:, 1]
            for j in range(nc):
                ph = p["phi%d_%d" % (j, k)]
                cols[:, nS + 1 + j] = np.cos(ph) * A[:, 2 + 2 * j] - np.sin(ph) * A[:, 3 + 2 * j]
            rows.append(cols)
            ys.append(sc["y"])
            ws.append(1.0 / np.sqrt(var))
            varl.append(var)
            colsl.append(cols)
        M = np.vstack(rows)
        y = np.concatenate(ys)
        w = np.concatenate(ws)
        if fixed is None:
            coef, *_ = np.linalg.lstsq(M * w[:, None], y * w, rcond=None)
        else:
            E, Aj = fixed
            yr = y - M[:, nS] * E - sum(M[:, nS + 1 + j] * Aj[j] for j in range(nc))
            cp, *_ = np.linalg.lstsq((M[:, :nS]) * w[:, None], yr * w, rcond=None)
            coef = np.concatenate([cp, [E], Aj])
        r = (y - M @ coef) * w
        return r, coef, varl, colsl

    def residuals(self, x, scans):
        r, _, varl, _ = self.joint(x, scans)
        if self.spec.get("random"):
            extra = [np.sqrt(np.maximum(np.log(v / sc["sig"] ** 2), 0.0)) for v, sc in zip(varl, scans)]
            return np.concatenate([r] + extra)
        return r


def fit_tied(tm, scans, base_res):
    """Warm start from the untied fit: shape params as fitted, phases from its quadratures (and
    from the opposite sign, since an amplitude's sign and its phase trade off)."""
    nS = len(scans)
    pb = tm.base.unpack(base_res.x, nS)
    names = tm.layout(nS)
    lo = np.array([-1200.0 if n.startswith("tau0") else (-4 * np.pi if n.startswith("phi")
                   else BOUNDS[n][0]) for n in names])
    hi = np.array([-900.0 if n.startswith("tau0") else (4 * np.pi if n.startswith("phi")
                   else BOUNDS[n][1]) for n in names])
    best = None
    for flip in (0.0, np.pi):
        x0 = list(base_res.x)
        for j in range(tm.n_comp()):
            for k, sc in enumerate(scans):
                c = tm.base.scan_fit(sc, pb["tau0_%d" % k], pb)[1]
                x0.append(np.arctan2(-c[3 + 2 * j], c[2 + 2 * j]) + flip * (k % 2))
        x0 = np.clip(np.array(x0, float), lo + 1e-9, hi - 1e-9)
        try:
            with np.errstate(all="ignore"):
                r = least_squares(tm.residuals, x0, args=(scans,), bounds=(lo, hi), x_scale="jac",
                                  max_nfev=4000)
        except (ValueError, np.linalg.LinAlgError):
            continue
        if np.isfinite(r.cost) and (best is None or r.cost < best.cost):
            best = r
    return best


def score_tied(tm, res, scans):
    nS = len(scans)
    r, coef, varl, colsl = tm.joint(res.x, scans)
    p = tm.unpack(res.x, nS)
    nll = float(r @ r) + sum(float(np.sum(np.log(v / sc["sig"] ** 2))) for v, sc in zip(varl, scans))
    n = sum(len(sc["t"]) for sc in scans)
    k_par = len(res.x) + len(coef)
    per, coefs, errs = {}, {}, {}
    off = 0
    M = np.vstack(colsl)
    w = np.concatenate([1 / np.sqrt(v) for v in varl])
    cov = np.linalg.pinv((M * w[:, None]).T @ (M * w[:, None])) * max(float(r @ r) / (n - M.shape[1]), 1.0)
    se = np.sqrt(np.diag(cov))
    for k, sc in enumerate(scans):
        rk = r[off:off + len(sc["t"])]
        off += len(sc["t"])
        win = np.abs(sc["t"] - p["tau0_%d" % k]) < 300
        per[sc["tag"]] = {"chi2": float(rk @ rk), "n": len(rk), "chi2_window": float(rk[win] @ rk[win]),
                          "n_window": int(win.sum())}
        c = [coef[k], coef[nS]]
        e = [se[k], se[nS]]
        for j in range(tm.n_comp()):
            ph, A = p["phi%d_%d" % (j, k)], coef[nS + 1 + j]
            c += [A * np.cos(ph), -A * np.sin(ph)]
            e += [se[nS + 1 + j]] * 2
        coefs[sc["tag"]] = [float(v) for v in c]
        errs[sc["tag"]] = [float(v) for v in e]
    return {"m2lnL": nll, "n": n, "k": k_par, "AIC": nll + 2 * k_par, "BIC": nll + k_par * np.log(n),
            "per_scan": per, "params": {kk: float(v) for kk, v in p.items()}, "coef": coefs,
            "coef_err": errs, "shared": {"E": float(coef[nS]), "E_err": float(se[nS]),
                                         "A": [float(v) for v in coef[nS + 1:]],
                                         "A_err": [float(v) for v in se[nS + 1:]]}}


def cv_tied(tm, scans, fits):
    """Hold the amplitudes AND shapes fitted on scan A; on scan B refit only its pedestal, tau0 and
    fringe phases. The strictest test that the two scans show the same physics."""
    out = {}
    for a in range(len(scans)):
        ra = fits.get(a)
        if ra is None:
            continue
        r1, coef1, _, _ = tm.joint(ra.x, [scans[a]])
        pa = tm.unpack(ra.x, 1)
        E, Aj = coef1[1], list(coef1[2:])
        for b in range(len(scans)):
            if b == a:
                continue
            nb = len(tm.base.layout(1))
            best = np.inf
            for t0 in np.arange(-1130, -990, 4.0):
                for ph0 in (0.0, np.pi / 2, np.pi, 3 * np.pi / 2):
                    x = np.array(list(ra.x[:nb]) + [ph0] * tm.n_comp(), float)
                    x[0] = t0

                    def res_b(z):
                        xx = x.copy()
                        xx[0] = z[0]
                        xx[nb:] = z[1:]
                        return tm.joint(xx, [scans[b]], fixed=(E, Aj))[0]
                    try:
                        rr = least_squares(res_b, np.array([t0] + [ph0] * tm.n_comp()), max_nfev=200)
                    except ValueError:
                        continue
                    best = min(best, 2 * rr.cost)
            out["{}->{}".format(scans[a]["tag"], scans[b]["tag"])] = {
                "m2lnL": best, "n": len(scans[b]["t"]), "per_pt": best / len(scans[b]["t"])}
    return out


def starts(model, scans, f):
    """Shape multi-starts at a given carrier frequency (only free parameters are varied)."""
    free = model.spec["free"]
    if isinstance(model, TemplateModel):
        out = []
        for kappa in ((0.9, 1.0, 1.12) if "kappa" in free else (1.0,)):
            for we in (90.0, 150.0):
                x = {"w_e": we, "dtau_e": 0.0, "kappa": kappa, "a": 0.3, "Ar": 0.8}
                for k in range(len(scans)):
                    x["tau0_%d" % k] = -1057.0
                out.append(np.array([x[n] for n in model.layout(len(scans))]))
        return out
    grid = {"w": (110.0, 180.0), "tail": (40.0, -40.0) if "tail" in free else (0.0,),
            "T": (100.0, 140.0) if "T" in free else (105.0,), "a": (0.3,), "Ar": (0.8,),
            "f2": model.spec.get("f2_starts", (0.00952,)) if "f2" in free else (0.00952,)}
    out = []
    for w in grid["w"]:
        for tail in grid["tail"]:
            for T in grid["T"]:
                for f2 in grid["f2"]:
                    x = {"w_f": w, "w_e": w, "dtau_e": 0.0, "f": f, "beta": 0.0, "tail": tail,
                         "r": 0.3, "T": T, "a": 0.3, "f2": f2, "Ar": 0.8}
                    for k in range(len(scans)):
                        x["tau0_%d" % k] = -1057.0
                    out.append(np.array([x[n] for n in model.layout(len(scans))]))
    return out


BOUNDS = {"w_f": (80, 500), "w_e": (80, 500), "dtau_e": (-150, 150), "f": (0.002, 0.26),
          "beta": (-2e-4, 2e-4), "tail": (-200, 200), "r": (-1.5, 1.5), "T": (60, 200),
          "a": (0.0, 0.9), "f2": (0.002, 0.26), "Ar": (0.0, 5.0), "kappa": (0.7, 1.4)}


def joint_profile(scans, fs, ws=np.arange(80, 401, 20.0), t0s=np.arange(-1130, -989, 3.0),
                  harmonics=(1,)):
    """Cheap chi2 profile vs carrier frequency (one shared Gaussian envelope; min over w and
    per-scan tau0), with fringe components at n*f for n in `harmonics`: used to find the
    frequency BASINS in which the full models are then optimised."""
    tot = np.zeros(len(fs))
    nc = 2 + 2 * len(harmonics)
    for sc in scans:
        t, y, sig = sc["t"], sc["y"], sc["sig"]
        yy = y @ y
        best = np.full(len(fs), np.inf)
        for w in ws:
            for t0 in t0s:
                g = K.gauss(t, t0, w)
                cols = [np.broadcast_to(np.ones_like(t), (len(fs), len(t))),
                        np.broadcast_to(g, (len(fs), len(t)))]
                for n in harmonics:
                    ph = 2 * np.pi * n * np.outer(fs, t - t0)
                    cols += [g * np.cos(ph), g * np.sin(ph)]
                M = np.empty((len(fs), nc, nc))
                b = np.stack([c @ y for c in cols], axis=1)
                for a_ in range(nc):
                    for c_ in range(a_, nc):
                        M[:, a_, c_] = M[:, c_, a_] = np.einsum("fn,fn->f", cols[a_], cols[c_])
                M += np.eye(nc) * 1e-9
                coef = np.linalg.solve(M, b[..., None])[..., 0]
                best = np.minimum(best, (yy - np.einsum("fa,fa->f", coef, b)) / sig ** 2)
        tot += best
    return tot


def basins(fs, prof, n_fast=10, n_slow=5):
    """Local minima of the profile, best first, each with its [lo, hi] bracket (adjacent maxima)."""
    out = []
    for region, n in ((fs > 0.1, n_fast), (fs < 0.05, n_slow)):
        idx = np.where(region)[0]
        f, p = fs[idx], prof[idx]
        mins = [i for i in range(1, len(p) - 1) if p[i] <= p[i - 1] and p[i] <= p[i + 1]]
        mins = sorted(mins, key=lambda i: p[i])[:n]
        for i in mins:
            lo = i
            while lo > 0 and p[lo - 1] >= p[lo]:
                lo -= 1
            hi = i
            while hi < len(p) - 1 and p[hi + 1] >= p[hi]:
                hi += 1
            out.append((float(f[i]), float(f[lo]), float(f[hi]), float(p[i])))
    return out


KAPPA_BASINS = {}


def template_profile(scans, kappas, sign, wes=(80.0, 110.0, 150.0, 200.0),
                     t0s=np.arange(-1130, -989, 2.0)):
    """chi2 of the kappa-free template model vs kappa (min over per-scan tau0 and w_e): the
    template's fringe phase is highly multimodal in kappa, so it is profiled on a fine grid."""
    m = TemplateModel("profile", {"free": ["kappa"], "sign": sign})
    tot = np.zeros(len(kappas))
    for sc in scans:
        best = np.full(len(kappas), np.inf)
        for i, kp in enumerate(kappas):
            for we in wes:
                for t0 in t0s:
                    r, _, _ = m.scan_fit(sc, t0, {"kappa": kp, "w_e": we, "dtau_e": 0.0, "a": 0.0,
                                                  "Ar": 0.0, "w_f": 0.0})
                    c2 = float(r @ r)
                    if c2 < best[i]:
                        best[i] = c2
        tot += best
    return tot


def basins_1d(x, p, n):
    """The n deepest local minima of a 1D profile, each with its bracket."""
    mins = [i for i in range(1, len(p) - 1) if p[i] <= p[i - 1] and p[i] <= p[i + 1]]
    mins = sorted(mins, key=lambda i: p[i])[:n]
    out = []
    for i in mins:
        lo = i
        while lo > 0 and p[lo - 1] >= p[lo]:
            lo -= 1
        hi = i
        while hi < len(p) - 1 and p[hi + 1] >= p[hi]:
            hi += 1
        out.append((float(x[i]), float(x[lo]), float(x[hi]), float(p[i])))
    return out


BASINS = None


def fit(model, scans, basin_list=None):
    """Best local fit over all frequency basins (f bounded to each) and shape multi-starts."""
    names = model.layout(len(scans))
    lo0 = np.array([-1200.0 if n.startswith("tau0") else BOUNDS[n][0] for n in names])
    hi0 = np.array([-900.0 if n.startswith("tau0") else BOUNDS[n][1] for n in names])
    blist = basin_list if basin_list is not None else model.spec.get("basins", BASINS)
    f_free = "f" in model.spec["free"]
    if isinstance(model, TemplateModel):
        blist, f_free = [(None, 0, 0, 0)], False
        if "kappa" in model.spec["free"] and KAPPA_BASINS.get(model.spec["sign"]):
            return fit_template_kappa(model, scans, lo0, hi0, names)
    best = None
    for (fc, flo, fhi, _) in (blist if (f_free or isinstance(model, TemplateModel))
                               else [(model.spec.get("f_fixed"), 0, 0, 0)]):
        for x0 in starts(model, scans, fc):
            lo, hi = lo0.copy(), hi0.copy()
            if f_free:
                j = names.index("f")
                lo[j], hi[j] = min(flo, fc - 1e-6), max(fhi, fc + 1e-6)
            x0 = np.clip(x0, lo + 1e-9, hi - 1e-9)
            try:
                with np.errstate(all="ignore"):
                    r = least_squares(model.residuals, x0, args=(scans,), bounds=(lo, hi),
                                      x_scale="jac", max_nfev=3000)
            except (ValueError, np.linalg.LinAlgError):
                continue
            if not np.isfinite(r.cost):
                continue
            if best is None or r.cost < best.cost:
                best = r
    return best


def fit_template_kappa(model, scans, lo0, hi0, names):
    """Local fits of a kappa-free template model inside each kappa basin (kappa bounded to it)."""
    j = names.index("kappa")
    best = None
    for (kc, klo, khi, _) in KAPPA_BASINS[model.spec["sign"]]:
        for x0 in starts(model, scans, None):
            lo, hi = lo0.copy(), hi0.copy()
            lo[j], hi[j] = min(klo, kc - 1e-6), max(khi, kc + 1e-6)
            x0[j] = kc
            x0 = np.clip(x0, lo + 1e-9, hi - 1e-9)
            try:
                with np.errstate(all="ignore"):
                    r = least_squares(model.residuals, x0, args=(scans,), bounds=(lo, hi),
                                      x_scale="jac", max_nfev=3000)
            except (ValueError, np.linalg.LinAlgError):
                continue
            if np.isfinite(r.cost) and (best is None or r.cost < best.cost):
                best = r
    return best


def score(model, res, scans):
    p = model.unpack(res.x, len(scans))
    nll, n = 0.0, 0
    per = {}
    coefs = {}
    errs = {}
    for k, sc in enumerate(scans):
        r, coef, var = model.scan_fit(sc, p["tau0_%d" % k], p)
        A, _ = model.design(sc["t"], p["tau0_%d" % k], p)
        Aw = A / np.sqrt(var)[:, None]
        cov = np.linalg.pinv(Aw.T @ Aw) * max(float(np.sum(r ** 2)) / max(len(r) - A.shape[1], 1), 1.0)
        errs[sc["tag"]] = np.sqrt(np.diag(cov)).tolist()
        c2 = float(np.sum(r ** 2))
        l = c2 + float(np.sum(np.log(var / sc["sig"] ** 2)))
        win = np.abs(sc["t"] - p["tau0_%d" % k]) < 300
        per[sc["tag"]] = {"chi2": c2, "n": len(r), "chi2_window": float(np.sum(r[win] ** 2)),
                          "n_window": int(win.sum())}
        coefs[sc["tag"]] = coef.tolist()
        nll += l
        n += len(r)
    k_par = len(res.x) + model.n_linear() * len(scans)
    return {"m2lnL": nll, "n": n, "k": k_par, "AIC": nll + 2 * k_par, "BIC": nll + k_par * np.log(n),
            "per_scan": per, "params": {kk: float(v) for kk, v in p.items()}, "coef": coefs,
            "coef_err": errs}


def alias_partners(f, step, fmax=0.26):
    """Frequencies sampled identically to f at this step: k/step +- f (the single-scan ambiguity)."""
    out = {f}
    for k in range(1, int(fmax * step) + 2):
        for g in (k / step - f, k / step + f):
            if 0.002 < g < fmax:
                out.add(g)
    return sorted(out)


def cross_validate(model, scans):
    """Shared physics fitted on scan A predicts scan B, refitting only B's tau0 and linear
    amplitudes. A single scan fixes the carrier only up to its aliasing partners, so every
    partner is tried and the best is kept -- the test is whether B is consistent with A at all."""
    out = {}
    for a in range(len(scans)):
        ra = fit(model, [scans[a]])
        if ra is None:
            continue
        pa = model.unpack(ra.x, 1)
        step = float(np.median(np.diff(scans[a]["t"])))
        if "f" in model.spec["free"] or "kappa" in model.spec["free"]:
            fset = alias_partners(pa["f"], step)
        else:
            fset = [pa["f"]]
        for b in range(len(scans)):
            if b == a:
                continue
            best = (np.inf, None)
            for f in fset:
                pb = dict(pa, f=f)
                if isinstance(model, TemplateModel):
                    pb["kappa"] = f * model.load_template()["T2"]
                for t0 in np.arange(-1150, -960, 2.0):
                    r, _, var = model.scan_fit(scans[b], t0, pb)
                    c2 = float(np.sum(r ** 2) + np.sum(np.log(var / scans[b]["sig"] ** 2)))
                    if c2 < best[0]:
                        best = (c2, f)
            out["{}->{}".format(scans[a]["tag"], scans[b]["tag"])] = {
                "m2lnL": best[0], "n": len(scans[b]["t"]), "per_pt": best[0] / len(scans[b]["t"]),
                "f_used": best[1]}
    return out


def lam(f):
    """Label a carrier: wavelength (nm) for optical carriers, period (fs) for slow ones."""
    return "{:.0f} nm".format(K.C_UM_PER_FS * 1e3 / f) if f > 0.1 else "{:.0f} fs".format(1 / f)


def load_scans():
    scans = []
    for tag in TAGS:
        d = K.load(tag)
        o = np.argsort(d["tau"])
        scans.append({"tag": tag, "t": d["tau"][o], "y": (d["X"] * K.ROT_DEG_PER_V)[o],
                      "sig": K.load_result(tag, "s1")["noise_per_point_deg"]})
    return scans


F_1620 = K.C_UM_PER_FS * 1e3 / 1620.0
SLOW = (1 / 62.0, 1 / 105.0, 1 / 140.0)

MODELS = [
    Model("M0 fringe at 1620 nm, one Gaussian", {"free": [], "f_fixed": F_1620}),
    Model("M1 carrier free", {"free": ["f"]}),
    Model("M2 +separate effect envelope", {"free": ["f"], "sep_env": True}),
    Model("M3 +chirp", {"free": ["f", "beta"], "sep_env": True}),
    Model("M4 +ring-down tail", {"free": ["f", "tail"], "sep_env": True}),
    Model("M5 +lock-in lag", {"free": ["f", "a"], "sep_env": True}),
    Model("M6 +cavity echoes", {"free": ["f", "r", "T"], "sep_env": True}),
    Model("M7 +2nd (slow) fringe", {"free": ["f", "f2"], "sep_env": True, "fringe2": True,
                                    "f2_starts": SLOW}),
    Model("M8 +partial coherence", {"free": ["f", "Ar"], "sep_env": True, "random": True}),
    Model("M9 2nd fringe + partial coh.", {"free": ["f", "f2", "Ar"], "sep_env": True,
                                          "fringe2": True, "random": True, "f2_starts": SLOW}),
    Model("M10 echoes + lock-in + partial coh.", {"free": ["f", "r", "T", "a", "Ar"],
                                                 "sep_env": True, "random": True}),
    Model("H1 fringes at w2 + 2 w2, 1620 nm fixed", {"free": [], "f_fixed": F_1620,
                                                     "sep_env": True, "harmonics": (1, 2)}),
    Model("H2 fringes at w2 + 2 w2, carrier free", {"free": ["f"], "sep_env": True,
                                                    "harmonics": (1, 2), "basins": "H"}),
    Model("M11 two free fast fringes", {"free": ["f", "f2"], "sep_env": True, "fringe2": True,
                                        "f2_starts": (F_1620, 1 / 5.33, 0.2098)}),
    TemplateModel("T1 simulated fringe, kappa = 1, +tau", {"free": [], "sign": 1.0}),
    TemplateModel("T2 simulated fringe, kappa = 1, -tau", {"free": [], "sign": -1.0}),
    TemplateModel("T3 simulated fringe, kappa free, +tau", {"free": ["kappa"], "sign": 1.0}),
    TemplateModel("T4 simulated fringe, kappa free, -tau", {"free": ["kappa"], "sign": -1.0}),
    TemplateModel("T5 sim fringe, kappa free, +partial coh.", {"free": ["kappa", "Ar"], "sign": 1.0,
                                                              "random": True}),
    TemplateModel("T6 sim fringe, kappa free, -tau, +partial coh.", {"free": ["kappa", "Ar"],
                                                                    "sign": -1.0, "random": True}),
]


def main():
    global BASINS
    scans = load_scans()
    fs = np.concatenate([np.arange(0.150, 0.240, 4e-5), 1.0 / np.arange(30.0, 400.0, 1.0)[::-1]])
    prof = joint_profile(scans, fs)
    BASINS = basins(fs, prof)
    ffast = fs[fs > 0.1]
    prof_h = joint_profile(scans, ffast, harmonics=(1, 2))
    hb = basins_1d(ffast, prof_h, 8)
    for m in MODELS:
        if m.spec.get("basins") == "H":
            m.spec["basins"] = hb
    print("harmonic-model basins:", ", ".join("{:.0f} nm ({:.0f})".format(
        K.C_UM_PER_FS * 1e3 / b[0], b[3]) for b in hb))
    i1620 = int(np.argmin(abs(ffast - F_1620)))
    print("harmonic-model profile at 1620 nm: {:.0f}; single-fringe profile at 1620 nm: {:.0f}".format(
        prof_h[i1620], prof[fs > 0.1][i1620]))
    print("frequency basins (best first):", ", ".join(
        "{:.0f} nm ({:.0f})".format(K.C_UM_PER_FS * 1e3 / b[0], b[3]) if b[0] > 0.1 else
        "{:.0f} fs ({:.0f})".format(1 / b[0], b[3]) for b in BASINS))
    np.savez(K.RESULTS / "combined" / "s7_profile.npz", fs=fs, prof=prof)
    kappas = np.arange(0.85, 1.25 + 1e-9, 0.001)
    kprof = {}
    for sign in (1.0, -1.0):
        kprof[sign] = template_profile(scans, kappas, sign)
        KAPPA_BASINS[sign] = basins_1d(kappas, kprof[sign], 6)
        print("template kappa basins ({:+.0f}):".format(sign), ", ".join(
            "{:.3f} ({:.0f})".format(b[0], b[3]) for b in KAPPA_BASINS[sign]))
    np.savez(K.RESULTS / "combined" / "s7_kappa_profile.npz", kappas=kappas, plus=kprof[1.0],
             minus=kprof[-1.0])
    results = []
    t_all = time.time()
    for m in MODELS:
        t0 = time.time()
        res = fit(m, scans)
        s = score(m, res, scans)
        s["cv"] = cross_validate(m, scans)
        s["name"] = m.name
        results.append((m, res, s))
        cv = "; ".join("{} {:.1f}/pt".format(k, v["per_pt"]) for k, v in s["cv"].items())
        extra = " f2 {:.0f} fs".format(1 / s["params"]["f2"]) if "f2" in m.spec["free"] else ""
        extra += " a {:.2f}".format(s["params"]["a"]) if "a" in m.spec["free"] else ""
        extra += " Ar {:.2f}".format(s["params"]["Ar"]) if "Ar" in m.spec["free"] else ""
        extra += " r {:.2f} T {:.0f}".format(s["params"]["r"], s["params"]["T"]) if "r" in m.spec["free"] else ""
        extra += " kappa {:.3f}".format(s["params"]["kappa"]) if "kappa" in s["params"] else ""
        print("{:36s} -2lnL {:7.1f} k {:2d} AIC {:7.1f} BIC {:7.1f} | 146 {:6.1f} 235 {:6.1f} | {}{} | "
              "CV {} | {:.0f}s".format(
                  m.name, s["m2lnL"], s["k"], s["AIC"], s["BIC"], s["per_scan"]["146"]["chi2"],
                  s["per_scan"]["235"]["chi2"], lam(s["params"]["f"]), extra, cv,
                  time.time() - t0),
              flush=True)
    base = {m.name.split()[0]: (m, res) for m, res, _ in results}
    for key in ("M2", "M7", "M11", "H1", "H2", "T4"):
        if key not in base or base[key][1] is None:
            continue
        m, res0 = base[key]
        tm = TiedModel(m)
        t0 = time.time()
        rt = fit_tied(tm, scans, res0)
        if rt is None:
            continue
        s_ = score_tied(tm, rt, scans)
        single = {}
        for a in range(len(scans)):
            ra = fit(m, [scans[a]])
            if ra is not None:
                single[a] = fit_tied(tm, [scans[a]], ra)
        s_["cv"] = cv_tied(tm, scans, single)
        s_["name"] = tm.name
        results.append((tm, rt, s_))
        sh = s_["shared"]
        cv = "; ".join("{} {:.1f}/pt".format(k, v["per_pt"]) for k, v in s_["cv"].items())
        print("{:36s} -2lnL {:7.1f} k {:2d} AIC {:7.1f} BIC {:7.1f} | 146 {:6.1f} 235 {:6.1f} | E {:+.2f}+-{:.2f} "
              "A {} | CV {} | {:.0f}s".format(tm.name, s_["m2lnL"], s_["k"], s_["AIC"], s_["BIC"],
                                             s_["per_scan"]["146"]["chi2"], s_["per_scan"]["235"]["chi2"],
                                             sh["E"], sh["E_err"],
                                             [round(abs(v), 2) for v in sh["A"]], cv, time.time() - t0),
              flush=True)
    K.save_result("combined", "s7", {"models": [s for _, _, s in results]})
    print("total {:.0f}s".format(time.time() - t_all))
    figures(scans, results, fs, prof, kappas, kprof)
    return scans, results


def figures(scans, results, fs, prof, kappas, kprof):
    plt = K.style()
    names = [s["name"] for _, _, s in results]
    bic = np.array([s["BIC"] for _, _, s in results])
    ib = int(np.argmin(bic))

    # ---- 1: the ladder ------------------------------------------------------------------ #
    fig, axs = plt.subplots(1, 2, figsize=(12, 0.42 * len(names) + 1.6), sharey=True)
    yy = np.arange(len(names))[::-1]
    axs[0].barh(yy, bic - bic.min(), color=[K.ORANGE if i == ib else K.BLUE for i in range(len(names))],
                height=0.6)
    axs[0].set_xscale("symlog", linthresh=10)
    axs[0].set_yticks(yy)
    axs[0].set_yticklabels(names, fontsize=8.5)
    axs[0].set_xlabel("ΔBIC vs best (lower is better)")
    axs[0].set_title("Model ladder: joint fit of scans 146 + 235")
    for y, (_, _, s) in zip(yy, results):
        axs[0].text(0.3, y, " χ²/pt {:.2f}".format(s["m2lnL"] / s["n"]), va="center", fontsize=7.5,
                    color=K.INK2)
    for i, key in enumerate(("146->235", "235->146")):
        v = [s["cv"].get(key, {}).get("per_pt", np.nan) for _, _, s in results]
        axs[1].plot(v, yy, marker="o", lw=0, ms=6, color=(K.BLUE, K.ORANGE)[i],
                    label="fit on {} → predict {}".format(*key.split("->")))
    axs[1].axvline(1.0, color=K.INK2, lw=0.8, ls=":")
    axs[1].set_xscale("log")
    axs[1].set_xlabel("cross-scan −2lnL per point (1 = noise)")
    axs[1].set_title("Does the model generalise to the other scan?")
    axs[1].legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    K.save(fig, "s7_ladder.png", "combined")

    # ---- 2: the best fits ---------------------------------------------------------------- #
    # curated: the best physically constrained model, the best self-consistent untied one, and the
    # simulated physics (tied) -- the raw BIC winner (M11) is unphysical (see README)
    want = ("M7t", "M7", "T4t")
    top = [next(i for i, (_, _, s) in enumerate(results) if s["name"].split()[0] == w)
           for w in want if any(s["name"].split()[0] == w for _, _, s in results)]
    fig, axs = plt.subplots(2 * len(top), 2, figsize=(12, 3.1 * len(top) * 2 * 0.62),
                            gridspec_kw={"height_ratios": [2.2, 1] * len(top)})
    for row, i in enumerate(top):
        m, res, s = results[i]
        p = m.unpack(res.x, len(scans))
        for col, sc in enumerate(scans):
            ax, axr = axs[2 * row, col], axs[2 * row + 1, col]
            t0 = p["tau0_%d" % col]
            A, _ = (m.base if isinstance(m, TiedModel) else m).design(sc["t"], t0, p)
            coef = np.array(s["coef"][sc["tag"]])
            win = np.abs(sc["t"] - t0) < 350
            ax.plot(sc["t"][win], sc["y"][win], color=K.BLUE, marker="o", ms=4, lw=0, label="measured")
            ax.plot(sc["t"][win], (A @ coef)[win], color=K.ORANGE, marker=".", ms=5, lw=1.2,
                    label="model (at the samples)")
            eff = coef[0] + coef[1] * A[:, 1]
            ax.plot(sc["t"][win], eff[win], color=K.AQUA, lw=1.4, label="pedestal + effect")
            ax.set_title("{} — scan {}: χ² {:.0f} / {} pts".format(m.name, sc["tag"],
                         s["per_scan"][sc["tag"]]["chi2"], s["per_scan"][sc["tag"]]["n"]), fontsize=9)
            ax.set_ylabel("θ (deg)")
            if row == 0 and col == 1:
                ax.legend(loc="upper right", fontsize=7.5)
            r = (sc["y"] - A @ coef) / sc["sig"]
            axr.bar(sc["t"][win], r[win], width=np.median(np.diff(sc["t"])) * 0.7, color=K.MAGENTA)
            axr.axhline(0, color=K.INK2, lw=0.8)
            axr.axhspan(-2, 2, color=K.GRID, alpha=0.6, lw=0)
            axr.set_ylabel("resid / σ")
    axs[-1, 0].set_xlabel("pump₂ delay τ (fs)")
    axs[-1, 1].set_xlabel("pump₂ delay τ (fs)")
    fig.tight_layout()
    K.footnote(fig)
    K.save(fig, "s7_best_fits.png", "combined")

    # ---- 3: carrier frequency and kappa ---------------------------------------------------- #
    fig, axs = plt.subplots(1, 3, figsize=(14, 4.2))
    fast = fs > 0.1
    axs[0].plot(K.C_UM_PER_FS * 1e3 / fs[fast], prof[fast], color=K.BLUE, lw=0.8)
    for lamm, lab in ((1620, "1620 delayed pump"), (1560, "1560"), (1605, "~1605 simulated")):
        axs[0].axvline(lamm, color=K.ORANGE if lamm == 1620 else K.INK2, lw=1.0, ls="--" if lamm == 1605 else "-")
        axs[0].text(lamm, prof[fast].max(), " " + lab, rotation=90, va="top", fontsize=7.5, color=K.INK2)
    axs[0].set_xlabel("fringe carrier λ (nm, apparent)")
    axs[0].set_ylabel("joint χ² (M1 model)")
    axs[0].set_title("fast-fringe hypothesis")
    slow = fs < 0.05
    axs[1].plot(1 / fs[slow], prof[slow], color=K.BLUE, lw=1.0)
    axs[1].set_xlabel("period (fs)")
    axs[1].set_title("slow-modulation hypothesis")
    axs[1].set_ylim(axs[0].get_ylim())
    for sign, colr in ((1.0, K.BLUE), (-1.0, K.ORANGE)):
        axs[2].plot(kappas, kprof[sign], color=colr, lw=1.0, label="delay sign {:+.0f}".format(sign))
    axs[2].axvline(1.0, color=K.INK2, lw=1.0)
    axs[2].set_xlabel("delay-axis scale κ")
    axs[2].set_ylabel("joint χ² (simulated template)")
    axs[2].set_title("simulated fringe: κ = 1 ⇔ the physics as simulated")
    axs[2].legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    K.save(fig, "s7_frequency.png", "combined")

    # ---- 4: model dependence of the effect ------------------------------------------------ #
    fig, ax = plt.subplots(figsize=(9, 0.42 * len(names) + 1.4))
    for j, sc in enumerate(scans):
        e = [s["coef"][sc["tag"]][1] for _, _, s in results]
        de = [s["coef_err"][sc["tag"]][1] for _, _, s in results]
        ax.errorbar(e, yy + (0.15 if j == 0 else -0.15), xerr=de, fmt="o", ms=4, capsize=2,
                    color=(K.BLUE, K.ORANGE)[j], label="scan {}".format(sc["tag"]))
    ax.axvline(0, color=K.INK2, lw=0.8)
    ax.set_yticks(yy)
    ax.set_yticklabels(names, fontsize=8.5)
    ax.set_xlabel("effect amplitude E (deg, k = 0 term)")
    ax.set_title("How much the 'effect' depends on the fringe model")
    ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    K.footnote(fig)
    K.save(fig, "s7_effect.png", "combined")


if __name__ == "__main__":
    main()
