import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# ============================================================
# SETTINGS
# ============================================================

RESULT_FILE = "loso_results.csv"
OUTPUT_DIR = Path("results")

OUTPUT_DIR.mkdir(exist_ok=True)

# ============================================================
# LOAD CSV
# ============================================================

print("=" * 70)
print("LOADING LOSO RESULTS")
print("=" * 70)

try:
    df = pd.read_csv(RESULT_FILE)
except FileNotFoundError:
    print(f"\nERROR: Could not find {RESULT_FILE}")
    print("Make sure loso_results.csv is in the FYP folder.")
    raise SystemExit

print("\nOriginal columns:")
print(df.columns.tolist())

print("\nFirst rows:")
print(df.head().to_string(index=False))

# ============================================================
# CLEAN COLUMN NAMES
# ============================================================

df.columns = (
    df.columns
    .str.strip()
    .str.lower()
    .str.replace("%", "", regex=False)
    .str.replace(" ", "_", regex=False)
)

print("\nCleaned columns:")
print(df.columns.tolist())

# ============================================================
# FIND COLUMNS AUTOMATICALLY
# ============================================================

def find_column(possible_names):

    for name in possible_names:
        if name in df.columns:
            return name

    return None


subject_col = find_column([
    "subject",
    "subjects",
    "target_subject",
    "target"
])

accuracy_col = find_column([
    "accuracy",
    "acc",
    "accuracy_%",
    "accuracy_percent"
])

precision_col = find_column([
    "precision",
    "prec",
    "precision_%",
    "precision_percent"
])

recall_col = find_column([
    "recall",
    "sensitivity",
    "recall_%",
    "recall_percent"
])

f1_col = find_column([
    "f1",
    "f1_score",
    "f1-score",
    "f1_%",
    "f1_percent"
])

kappa_col = find_column([
    "kappa",
    "cohen_kappa",
    "cohen's_kappa"
])

print("\nDetected columns:")
print("Subject  :", subject_col)
print("Accuracy :", accuracy_col)
print("Precision:", precision_col)
print("Recall   :", recall_col)
print("F1       :", f1_col)
print("Kappa    :", kappa_col)

# ============================================================
# VALIDATE REQUIRED COLUMNS
# ============================================================

required = {
    "Subject": subject_col,
    "Accuracy": accuracy_col,
    "Precision": precision_col,
    "Recall": recall_col,
    "F1": f1_col,
    "Kappa": kappa_col
}

missing = [
    name
    for name, column in required.items()
    if column is None
]

if missing:

    print("\n" + "=" * 70)
    print("ERROR: REQUIRED COLUMNS NOT FOUND")
    print("=" * 70)

    print("\nMissing:")
    for item in missing:
        print(" -", item)

    print("\nAvailable columns:")
    for column in df.columns:
        print(" -", column)

    print("\nPlease send me this output if the script stops here.")

    raise SystemExit

# ============================================================
# CONVERT NUMERIC COLUMNS
# ============================================================

metric_columns = [
    accuracy_col,
    precision_col,
    recall_col,
    f1_col,
    kappa_col
]

for column in metric_columns:

    df[column] = (
        df[column]
        .astype(str)
        .str.replace("%", "", regex=False)
        .str.strip()
    )

    df[column] = pd.to_numeric(
        df[column],
        errors="coerce"
    )

# ============================================================
# CONVERT PERCENTAGES
# ============================================================

# Accuracy / Precision / Recall / F1
# should normally be either:
#
# 0.785 -> 78.5%
#
# or
#
# 78.5 -> 78.5%

for column in [
    accuracy_col,
    precision_col,
    recall_col,
    f1_col
]:

    if df[column].max() <= 1.0:
        df[column] = df[column] * 100

# ============================================================
# DISPLAY CLEAN DATA
# ============================================================

print("\n" + "=" * 70)
print("CLEAN LOSO RESULTS")
print("=" * 70)

display_df = pd.DataFrame({
    "Subject": df[subject_col],
    "Accuracy": df[accuracy_col],
    "Precision": df[precision_col],
    "Recall": df[recall_col],
    "F1": df[f1_col],
    "Kappa": df[kappa_col]
})

print(display_df.to_string(index=False))

# ============================================================
# SUMMARY STATISTICS
# ============================================================

summary = pd.DataFrame({
    "Metric": [
        "Accuracy",
        "Precision",
        "Recall",
        "F1",
        "Kappa"
    ],

    "Mean": [
        df[accuracy_col].mean(),
        df[precision_col].mean(),
        df[recall_col].mean(),
        df[f1_col].mean(),
        df[kappa_col].mean()
    ],

    "Std": [
        df[accuracy_col].std(),
        df[precision_col].std(),
        df[recall_col].std(),
        df[f1_col].std(),
        df[kappa_col].std()
    ]
})

print("\n" + "=" * 70)
print("MEAN ± STANDARD DEVIATION")
print("=" * 70)

for _, row in summary.iterrows():

    print(
        f"{row['Metric']:10s}: "
        f"{row['Mean']:.2f} ± {row['Std']:.2f}"
    )

summary.to_csv(
    OUTPUT_DIR / "summary_statistics.csv",
    index=False
)

# ============================================================
# SUBJECT LABELS
# ============================================================

subjects = df[subject_col].astype(str).values

# ============================================================
# 1. ACCURACY
# ============================================================

plt.figure(figsize=(10, 6))

values = df[accuracy_col].values

bars = plt.bar(
    subjects,
    values,
    color="steelblue",
    edgecolor="black"
)

mean_value = values.mean()

plt.axhline(
    mean_value,
    color="red",
    linestyle="--",
    linewidth=2,
    label=f"Mean = {mean_value:.2f}%"
)

for bar, value in zip(bars, values):

    plt.text(
        bar.get_x() + bar.get_width() / 2,
        value + 1,
        f"{value:.1f}%",
        ha="center",
        va="bottom",
        fontsize=9
    )

plt.xlabel("Target Subject")
plt.ylabel("Accuracy (%)")
plt.title("LOSO Accuracy Across Subjects")
plt.ylim(0, 100)
plt.grid(axis="y", alpha=0.3)
plt.legend()

plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "loso_accuracy.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()

# ============================================================
# 2. F1 SCORE
# ============================================================

plt.figure(figsize=(10, 6))

values = df[f1_col].values

bars = plt.bar(
    subjects,
    values,
    color="seagreen",
    edgecolor="black"
)

mean_value = values.mean()

plt.axhline(
    mean_value,
    color="red",
    linestyle="--",
    linewidth=2,
    label=f"Mean = {mean_value:.2f}%"
)

for bar, value in zip(bars, values):

    plt.text(
        bar.get_x() + bar.get_width() / 2,
        value + 1,
        f"{value:.1f}%",
        ha="center",
        va="bottom",
        fontsize=9
    )

plt.xlabel("Target Subject")
plt.ylabel("F1 Score (%)")
plt.title("LOSO F1 Score Across Subjects")
plt.ylim(0, 100)
plt.grid(axis="y", alpha=0.3)
plt.legend()

plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "loso_f1.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()

# ============================================================
# 3. KAPPA
# ============================================================

plt.figure(figsize=(10, 6))

values = df[kappa_col].values

bars = plt.bar(
    subjects,
    values,
    color="darkorange",
    edgecolor="black"
)

mean_value = values.mean()

plt.axhline(
    mean_value,
    color="red",
    linestyle="--",
    linewidth=2,
    label=f"Mean = {mean_value:.3f}"
)

for bar, value in zip(bars, values):

    plt.text(
        bar.get_x() + bar.get_width() / 2,
        value + 0.02,
        f"{value:.3f}",
        ha="center",
        va="bottom",
        fontsize=9
    )

plt.xlabel("Target Subject")
plt.ylabel("Cohen's Kappa")
plt.title("LOSO Cohen's Kappa Across Subjects")
plt.ylim(0, 1)
plt.grid(axis="y", alpha=0.3)
plt.legend()

plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "loso_kappa.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()

# ============================================================
# 4. ALL METRICS
# ============================================================

plt.figure(figsize=(12, 7))

x = np.arange(len(subjects))
width = 0.2

plt.bar(
    x - 1.5 * width,
    df[accuracy_col],
    width,
    label="Accuracy"
)

plt.bar(
    x - 0.5 * width,
    df[precision_col],
    width,
    label="Precision"
)

plt.bar(
    x + 0.5 * width,
    df[recall_col],
    width,
    label="Recall"
)

plt.bar(
    x + 1.5 * width,
    df[f1_col],
    width,
    label="F1"
)

plt.xticks(
    x,
    subjects
)

plt.xlabel("Target Subject")
plt.ylabel("Score (%)")
plt.title("LOSO Performance Across Subjects")
plt.ylim(0, 100)

plt.legend()
plt.grid(axis="y", alpha=0.3)

plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "all_metrics.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()

# ============================================================
# 5. SAVE CLEAN RESULTS
# ============================================================

display_df.to_csv(
    OUTPUT_DIR / "clean_loso_results.csv",
    index=False
)

# ============================================================
# DONE
# ============================================================

print("\n" + "=" * 70)
print("PLOTTING COMPLETE")
print("=" * 70)

print("\nFiles created:")

for file in sorted(OUTPUT_DIR.iterdir()):
    print(" -", file)

print("\nYou can now open the PNG files from the results folder.")