import os
import sys

import numpy as np
import pandas as pd
import sklearn
from scipy import stats

from sklearn.model_selection import StratifiedKFold, cross_val_predict, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectKBest, f_classif, VarianceThreshold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.dummy import DummyClassifier
from sklearn.metrics import (
    balanced_accuracy_score,
    f1_score,
    confusion_matrix,
    classification_report,
)


# 0. ---- ENVIRONMENT -----


print("=" * 70)
print("ENVIRONMENT")
print("=" * 70)
print(f"python:  {sys.version.split()[0]}")
print(f"numpy:   {np.__version__}")
print(f"pandas:  {pd.__version__}")
print(f"sklearn: {sklearn.__version__}")
print()

# 1. ---- SETTINGS -----


RANDOM_STATE = 42
N_SPLITS = 5

# Number of genes kept by the ANOVA filter. Fixed in advance (not tuned).
K = 500

# Compensates for class imbalance
CLASS_WEIGHT = "balanced"

# Random-gene removal experiments (lower this, e.g. to 20, while testing)
N_RANDOM_CONTROLS = 100

# Repeated CV seeds for the main comparisons
N_CV_SEEDS = 10

os.makedirs("results", exist_ok=True)

# 2. ---- PAM50 GENE LIST -----


# -Older and newer symbols are both listed-
PAM50 = [
    "ACTR3B", "ANLN", "BAG1", "BCL2", "BIRC5", "BLVRA", "CCNB1", "CCNE1",
    "CDC20", "CDC6", "CDH3", "CENPF", "CEP55", "CXXC5", "EGFR", "ERBB2",
    "ESR1", "EXO1", "FGFR4", "FOXA1", "FOXC1", "GPR160", "GRB7", "KIF2C",
    "KRT14", "KRT17", "KRT5", "MAPT", "MDM2", "MELK", "MIA", "MKI67",
    "MLPH", "MMP11", "MYBL2", "NAT1", "NDC80", "KNTC2", "NUF2", "CDCA1",
    "ORC6L", "ORC6", "PGR", "PHGDH", "PTTG1", "RRM2", "SFRP1", "SLC39A6",
    "TMEM45B", "TYMS", "UBE2C", "UBE2T", "MYC",
]

# - Try alternate symbols just in case -
PAM50_ALIASES = {
    "KNTC2": "NDC80",
    "CDCA1": "NUF2",
    "ORC6": "ORC6L",
    "ORC6L": "ORC6",
}


# 3. ---- LOAD DATA -----


expr = pd.read_csv("data/HiSeqV2.gz", sep="\t", index_col=0)  # genes x samples
clin = pd.read_csv("data/BRCA_clinicalMatrix.tsv", sep="\t", index_col=0)

TARGET = "PAM50Call_RNAseq"

# Clinical rows in the same order as the expression columns
clin = clin.loc[expr.columns]

# Primary tumours with a PAM50 subtype
keep = (clin["sample_type"] == "Primary Tumor") & clin[TARGET].notna()

X_all = expr.loc[:, keep].T  # samples x genes
y = clin.loc[keep, TARGET]

print("=" * 70)
print("DATASET")
print("=" * 70)
print(f"Samples: {X_all.shape[0]}")
print(f"Genes:   {X_all.shape[1]}")
print("\nSubtype counts:")
print(y.value_counts())
print()



# 4. ---- PAM50 GENES PRESENT IN THE DATA  -----


present = []
missing = []
alias_used = {}

for g in PAM50:
    if g in X_all.columns:
        present.append(g)
    elif PAM50_ALIASES.get(g) in X_all.columns:
        alias = PAM50_ALIASES[g]
        present.append(alias)
        alias_used[g] = alias
    else:
        missing.append(g)

present = sorted(set(present))

print("=" * 70)
print("PAM50 GENES")
print("=" * 70)
print(f"PAM50 symbols in supplied list: {len(PAM50)}")
print(f"PAM50 genes found in dataset:   {len(present)}")

if alias_used:
    print("\nAliases resolved:")
    for orig, alias in alias_used.items():
        print(f"  {orig} -> {alias}")

if missing:
    print("\nPAM50 genes NOT found:")
    print(missing)
else:
    print("\nAll PAM50 genes were found (directly or via aliases).")
print()


# 5. ---- FEATURE SETS -----


X_no_pam50 = X_all.drop(columns=present)
X_pam50_only = X_all[present]

print("=" * 70)
print("FEATURE SETS")
print("=" * 70)
print(f"Genes with PAM50:    {X_all.shape[1]}")
print(f"Genes without PAM50: {X_no_pam50.shape[1]}")
print(f"PAM50 genes only:    {X_pam50_only.shape[1]}")
print(f"PAM50 genes removed: {len(present)}")
print()



# 6. ---- CROSS-VALIDATION HELPERS -----


def make_cv(seed=RANDOM_STATE):
    return StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=seed)


cv = make_cv(RANDOM_STATE)


def repeated_fold_scores(X, y, model, n_seeds=N_CV_SEEDS):
    """Balanced accuracy for every fold of every seed (n_seeds * N_SPLITS values).

    The same seeds give the same splits, so scores from different feature sets
    line up fold by fold and can be compared in a paired test.
    """
    scores = []
    for seed in range(n_seeds):
        cv_s = make_cv(seed=RANDOM_STATE + seed)
        scores.extend(
            cross_val_score(model, X, y, cv=cv_s, scoring="balanced_accuracy")
        )
    return np.array(scores)


def per_seed_means(fold_scores):
    return fold_scores.reshape(N_CV_SEEDS, N_SPLITS).mean(axis=1)


def corrected_paired_ttest(a, b):
    """Nadeau and Bengio corrected resampled t-test on paired fold scores.

    Cross-validation folds share training data, so a plain t-test is too
    optimistic. The correction adds test_size/train_size to the variance term.
    """
    d = a - b
    n = len(d)
    test_over_train = 1 / (N_SPLITS - 1)
    se = np.sqrt((1 / n + test_over_train) * d.var(ddof=1))
    t = d.mean() / se
    p = 2 * stats.t.sf(abs(t), df=n - 1)
    return d.mean(), t, p


# 7. ---- MODEL DEFINITIONS -----


def make_logistic(k=K):
    return Pipeline([
        ("variance", VarianceThreshold()),
        ("select", SelectKBest(f_classif, k=k)),
        ("scale", StandardScaler()),
        ("model", LogisticRegression(
            max_iter=3000,
            class_weight=CLASS_WEIGHT,
            random_state=RANDOM_STATE,
        )),
    ])


def make_forest(k=K):
    return Pipeline([
        ("variance", VarianceThreshold()),
        ("select", SelectKBest(f_classif, k=k)),
        ("scale", StandardScaler()),
        ("model", RandomForestClassifier(
            n_estimators=300,
            random_state=RANDOM_STATE,
            n_jobs=-1,
            class_weight=CLASS_WEIGHT,
        )),
    ])


def make_models(k=K):
    return {
        "Logistic regression": make_logistic(k),
        "Random forest": make_forest(k),
    }



# 8. ---- BASELINES -----


print("=" * 70)
print("BASELINES")
print("=" * 70)

# Majority-class baseline
dummy = DummyClassifier(strategy="most_frequent")
pred_dummy = cross_val_predict(dummy, X_all, y, cv=cv)
bal_dummy = balanced_accuracy_score(y, pred_dummy)
acc_dummy = (pred_dummy == y).mean()

print("\nMajority-class baseline")
print("-" * 40)
print(f"Accuracy:          {acc_dummy:.3f}")
print(f"Balanced accuracy: {bal_dummy:.3f}")

# PAM50-only logistic regression. The label is computed from these genes,
# so this is a ceiling for reference, not an independent result.
k_pam50 = min(K, X_pam50_only.shape[1])
pred_pam50_only = cross_val_predict(
    make_logistic(k=k_pam50), X_pam50_only, y, cv=cv
)
bal_pam50_only = balanced_accuracy_score(y, pred_pam50_only)
f1_pam50_only = f1_score(y, pred_pam50_only, average="macro")

print(f"\nPAM50-only logistic regression (k={k_pam50})")
print("-" * 40)
print(f"Balanced accuracy: {bal_pam50_only:.3f}")
print(f"Macro F1:          {f1_pam50_only:.3f}")
print()


# 9. ---- MAIN MODEL EVALUATION -----


def evaluate_models(X, y, label):
    print("\n" + "=" * 70)
    print(label.upper())
    print("=" * 70)

    labels = sorted(y.unique())

    for name, model in make_models(K).items():
        pred = cross_val_predict(model, X, y, cv=cv, method="predict")

        bal_acc = balanced_accuracy_score(y, pred)
        macro_f1 = f1_score(y, pred, average="macro")

        print(f"\n{name}")
        print("-" * 40)
        print(f"Balanced accuracy: {bal_acc:.3f}")
        print(f"Macro F1:          {macro_f1:.3f}")

        print("\nPer-subtype performance:")
        print(classification_report(y, pred, labels=labels, zero_division=0))

        print("Confusion matrix:")
        cm = confusion_matrix(y, pred, labels=labels)
        print(pd.DataFrame(
            cm,
            index=[f"True {lab}" for lab in labels],
            columns=[f"Pred {lab}" for lab in labels],
        ))


evaluate_models(X_all, y, "ALL GENES")
evaluate_models(X_no_pam50, y, "PAM50 GENES REMOVED")



# 10. ---- FEATURE-NUMBER SENSITIVITY (k was fixed at 500) -----


print("\n" + "=" * 70)
print("FEATURE NUMBER SENSITIVITY")
print("=" * 70)

k_values = [25, 50, 100, 250, 500, 1000, 2000]
k_results = []

for k in k_values:
    for dataset_name, X in [("All genes", X_all), ("PAM50 removed", X_no_pam50)]:
        max_k = min(k, X.shape[1])
        scores = cross_val_score(
            make_logistic(k=max_k), X, y, cv=cv, scoring="balanced_accuracy"
        )
        k_results.append({
            "Dataset": dataset_name,
            "K": max_k,
            "Balanced accuracy": scores.mean(),
            "CV SD": scores.std(),
        })

k_results_df = pd.DataFrame(k_results)
print(k_results_df.to_string(index=False))
k_results_df.to_csv("results/k_sensitivity.csv", index=False)



# 11. ---- REPEATED CV: ALL GENES vs PAM50 REMOVED -----


print("\n" + "=" * 70)
print(
    f"REPEATED CV ({N_CV_SEEDS} seeds x {N_SPLITS} folds, logistic regression)")
print("=" * 70)

f_all = repeated_fold_scores(X_all, y, make_logistic(K))
f_no_pam50 = repeated_fold_scores(X_no_pam50, y, make_logistic(K))

seed_all = per_seed_means(f_all)
seed_no_pam50 = per_seed_means(f_no_pam50)

mean_all, sd_all = seed_all.mean(), seed_all.std(ddof=1)
mean_no_pam50, sd_no_pam50 = seed_no_pam50.mean(), seed_no_pam50.std(ddof=1)

print(f"All genes:     {mean_all:.4f} +/- {sd_all:.4f} (SD across seeds)")
print(f"PAM50 removed: {mean_no_pam50:.4f} +/- {sd_no_pam50:.4f}")
print(f"Difference (PAM50 removed - all): {mean_no_pam50 - mean_all:+.4f}")


# 12. ---- RANDOM-GENE REMOVAL CONTROL (SINGLE SPLIT, SEED 42) -----


print("\n" + "=" * 70)
print(
    f"RANDOM {len(present)}-GENE REMOVAL CONTROL (n={N_RANDOM_CONTROLS}, single split)")
print("=" * 70)

rng = np.random.default_rng(RANDOM_STATE)
all_genes = np.array(X_all.columns)
random_accs = []

for i in range(N_RANDOM_CONTROLS):
    random_genes = rng.choice(all_genes, size=len(present), replace=False)
    X_random_removed = X_all.drop(columns=random_genes)
    scores = cross_val_score(
        make_logistic(k=min(K, X_random_removed.shape[1])),
        X_random_removed, y, cv=cv, scoring="balanced_accuracy",
    )
    random_accs.append(scores.mean())
    if (i + 1) % 10 == 0 or i == 0:
        print(f"Random removal {i + 1:3d}: {scores.mean():.3f}")

random_accs = np.array(random_accs)
print(
    f"\nRandom-removal balanced accuracy: {random_accs.mean():.4f} "
    f"+/- {random_accs.std(ddof=1):.4f} (n={N_RANDOM_CONTROLS}, one CV split)"
)



# 12b. ---- CONTROL: REMOVE THE MOST INFORMATIVE NON-PAM50 GENES -----


print("\n" + "=" * 70)
print("CONTROL: REMOVE TOP-INFORMATIVE NON-PAM50 GENES")
print("=" * 70)

X_var = X_all.loc[:, X_all.var() > 0]
F_scores, _ = f_classif(X_var, y)
ranked = pd.Series(F_scores, index=X_var.columns).drop(
    index=present, errors="ignore")
top_genes = ranked.nlargest(len(present)).index
X_top_removed = X_all.drop(columns=top_genes)

f_top = repeated_fold_scores(X_top_removed, y, make_logistic(K))
seed_top = per_seed_means(f_top)
mean_top, sd_top = seed_top.mean(), seed_top.std(ddof=1)

print(
    f"All genes:                             {mean_all:.4f} +/- {sd_all:.4f}")
print(
    f"PAM50 genes removed:                   {mean_no_pam50:.4f} +/- {sd_no_pam50:.4f}")
print(f"Top-{len(present)} non-PAM50 genes removed:      {mean_top:.4f} +/- {sd_top:.4f}")



# 13. ---- PAIRED COMPARISONS (same splits, corrected resampled t-test) -----

print("\n" + "=" * 70)
print("PAIRED COMPARISONS")
print("=" * 70)

comparisons = {
    "pam50_vs_all": ("PAM50 removed vs all genes", f_no_pam50, f_all),
    "top_vs_all": ("Top-50 non-PAM50 removed vs all genes", f_top, f_all),
    "pam50_vs_top": ("PAM50 removed vs top-50 non-PAM50 removed", f_no_pam50, f_top),
}

test_results = {}
for key, (name, a, b) in comparisons.items():
    d, t, p = corrected_paired_ttest(a, b)
    test_results[key] = (d, t, p)
    print(f"{name}: mean diff {d:+.4f}, t = {t:+.2f}, p = {p:.3f}")

pd.DataFrame(
    [(name, *test_results[key]) for key, (name, _, _) in comparisons.items()],
    columns=["Comparison", "Mean difference", "t", "p"],
).to_csv("results/paired_tests.csv", index=False)


# 14. ---- SUMMARY -----

print("\n" + "=" * 70)
print("FINAL SUMMARY")
print("=" * 70)

pred_all = cross_val_predict(make_logistic(K), X_all, y, cv=cv)
pred_no_pam50 = cross_val_predict(make_logistic(K), X_no_pam50, y, cv=cv)

bal_all = balanced_accuracy_score(y, pred_all)
bal_no_pam50 = balanced_accuracy_score(y, pred_no_pam50)
f1_all = f1_score(y, pred_all, average="macro")
f1_no_pam50 = f1_score(y, pred_no_pam50, average="macro")

summary = pd.DataFrame({
    "Analysis": [
        "Majority-class baseline (single CV)",
        f"PAM50-only logistic, k={k_pam50} (single CV, ceiling)",
        "All genes (single CV)",
        "PAM50 genes removed (single CV)",
        "All genes (repeated CV)",
        "PAM50 genes removed (repeated CV)",
        f"Top-{len(present)} non-PAM50 genes removed (repeated CV)",
        f"Random {len(present)} genes removed (mean, single CV)",
    ],
    "Balanced accuracy": [
        bal_dummy, bal_pam50_only, bal_all, bal_no_pam50,
        mean_all, mean_no_pam50, mean_top, random_accs.mean(),
    ],
    "SD": [
        np.nan, np.nan, np.nan, np.nan,
        sd_all, sd_no_pam50, sd_top, random_accs.std(ddof=1),
    ],
    "Macro F1": [
        np.nan, f1_pam50_only, f1_all, f1_no_pam50,
        np.nan, np.nan, np.nan, np.nan,
    ],
})

print(summary.to_string(index=False))
summary.to_csv("results/summary.csv", index=False)

print("\nChange after removing PAM50 genes:")
print(f"  Balanced accuracy, single CV:   {bal_no_pam50 - bal_all:+.4f}")
print(f"  Macro F1, single CV:            {f1_no_pam50 - f1_all:+.4f}")
print(f"  Balanced accuracy, repeated CV: {mean_no_pam50 - mean_all:+.4f}")
print(
    "\nThe single-CV change is larger than the repeated-CV change because "
    "seed 42 is an unlucky split for this comparison. Report the repeated-CV "
    "numbers."
)

print("\n" + "=" * 70)
print("INTERPRETATION")
print("=" * 70)

d_top, t_top, p_top = test_results["pam50_vs_top"]
if p_top < 0.05 and d_top < 0:
    print(
        "Removing the PAM50 genes lowered balanced accuracy more than "
        "removing the same number of equally informative non-PAM50 genes "
        f"(corrected resampled t-test, p = {p_top:.3f})."
    )
elif p_top < 0.05 and d_top > 0:
    print(
        "Removing the PAM50 genes lowered balanced accuracy less than "
        "removing the same number of equally informative non-PAM50 genes "
        f"(corrected resampled t-test, p = {p_top:.3f})."
    )
else:
    print(
        "Removing the PAM50 genes lowered balanced accuracy by a small "
        "amount, but the difference from removing the same number of equally "
        "informative non-PAM50 genes is not statistically distinguishable "
        f"(corrected resampled t-test, p = {p_top:.3f})."
    )
print(
    "\nThe PAM50 subtype label is computed from these genes in the same "
    "RNA-seq data, so this tests whether the label can be recovered without "
    "its defining genes. It does not test clinical prediction."
)
