import csv

input_file = "sic_pecvd_1hour40min_test.txt"
output_file = "sic.csv"

with open(input_file, "r") as txt, open(output_file, "w", newline="") as csv_out:
    writer = csv.writer(csv_out)
    writer.writerow(["wavelength", "n", "k"])  # Add header
    for i, line in enumerate(txt):
        if (i < 3):
            continue
        row = line.split()
        if row:
            writer.writerow(row)

print(f"Done! Saved to {output_file}")
