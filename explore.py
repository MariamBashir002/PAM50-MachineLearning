import pandas as pd

# rows are genes, columns are samples
expr = pd.read_csv("data/HiSeqV2.gz", sep="\t", index_col=0)
clin = pd.read_csv("data/BRCA_clinicalMatrix.tsv", sep="\t", index_col=0)

print("expression (genes, samples):", expr.shape)
print("clinical (samples, columns):", clin.shape)
print("samples in both:", len(set(expr.columns) & set(clin.index)), "\n")

if "sample_type" in clin.columns:
    print(clin["sample_type"].value_counts(dropna=False), "\n")

pam = [c for c in clin.columns if "PAM50" in c]
print("PAM50 columns:", pam, "\n")
for c in pam:
    print(clin[c].value_counts(dropna=False), "\n")
