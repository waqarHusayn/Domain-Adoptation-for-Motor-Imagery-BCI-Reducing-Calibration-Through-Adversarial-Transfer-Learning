import os

import numpy as np
import torch
import torch.nn as nn

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    cohen_kappa_score
)

from dataset import load_subject
from model import AdversarialEEGNet


# ============================================================
# CONFIGURATION
# ============================================================

DATA_DIR = "BCICIV_2b_gdf"

MODEL_PATH = "adversarial_eegnet.pth"

TARGET_SUBJECT = 1

BATCH_SIZE = 32


# ============================================================
# DEVICE
# ============================================================

device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

print("=" * 70)
print("ADVERSARIAL EEGNET EVALUATION")
print("=" * 70)

print(
    "Device:",
    device
)


# ============================================================
# LOAD TARGET SUBJECT
# ============================================================

print("\n")
print("=" * 70)
print(
    f"LOADING TARGET SUBJECT {TARGET_SUBJECT:02d}"
)
print("=" * 70)

X, y = load_subject(
    TARGET_SUBJECT,
    DATA_DIR
)


print("\nTarget data:")
print("X:", X.shape)
print("y:", y.shape)


# ============================================================
# CONVERT TO PYTORCH
# ============================================================

X = torch.tensor(
    X,
    dtype=torch.float32
).unsqueeze(1)

y_tensor = torch.tensor(
    y,
    dtype=torch.long
)


# ============================================================
# LOAD CHECKPOINT
# ============================================================

print("\n")
print("=" * 70)
print("LOADING MODEL")
print("=" * 70)

checkpoint = torch.load(
    MODEL_PATH,
    map_location=device
)


source_subjects = checkpoint[
    "source_subjects"
]

target_subject = checkpoint[
    "target_subject"
]

best_val_accuracy = checkpoint[
    "best_val_accuracy"
]

saved_epoch = checkpoint[
    "epoch"
]


print(
    "Source subjects:",
    source_subjects
)

print(
    "Target subject:",
    target_subject
)

print(
    "Best validation accuracy:",
    f"{best_val_accuracy:.4f}"
)

print(
    "Saved at epoch:",
    saved_epoch
)


# ============================================================
# CREATE MODEL
# ============================================================

model = AdversarialEEGNet(
    channels=3,
    samples=1000,
    num_classes=2,
    num_domains=len(source_subjects),
    lambda_value=0.0
).to(device)


model.load_state_dict(
    checkpoint[
        "model_state_dict"
    ]
)

model.eval()


# ============================================================
# PREDICTION
# ============================================================

all_predictions = []

all_probabilities = []

batch_count = int(
    np.ceil(
        len(X) / BATCH_SIZE
    )
)


print("\n")
print("=" * 70)
print("RUNNING TARGET EVALUATION")
print("=" * 70)


with torch.no_grad():

    for start in range(
        0,
        len(X),
        BATCH_SIZE
    ):

        end = min(
            start + BATCH_SIZE,
            len(X)
        )

        batch = X[
            start:end
        ].to(device)

        class_output, _ = model(
            batch
        )

        probabilities = torch.softmax(
            class_output,
            dim=1
        )

        predictions = (
            class_output
            .argmax(dim=1)
            .cpu()
            .numpy()
        )

        probabilities = (
            probabilities
            .cpu()
            .numpy()
        )

        all_predictions.extend(
            predictions
        )

        all_probabilities.extend(
            probabilities
        )


predictions = np.asarray(
    all_predictions
)

probabilities = np.asarray(
    all_probabilities
)


# ============================================================
# METRICS
# ============================================================

accuracy = accuracy_score(
    y,
    predictions
)

precision = precision_score(
    y,
    predictions,
    average="binary",
    zero_division=0
)

recall = recall_score(
    y,
    predictions,
    average="binary",
    zero_division=0
)

f1 = f1_score(
    y,
    predictions,
    average="binary",
    zero_division=0
)

kappa = cohen_kappa_score(
    y,
    predictions
)

cm = confusion_matrix(
    y,
    predictions
)


# ============================================================
# RESULTS
# ============================================================

print("\n")
print("=" * 70)
print("TARGET SUBJECT RESULTS")
print("=" * 70)

print(
    f"Target subject: {TARGET_SUBJECT:02d}"
)

print(
    f"Accuracy:  {accuracy:.4f} "
    f"({accuracy * 100:.2f}%)"
)

print(
    f"Precision: {precision:.4f}"
)

print(
    f"Recall:    {recall:.4f}"
)

print(
    f"F1-score:  {f1:.4f}"
)

print(
    f"Kappa:     {kappa:.4f}"
)


# ============================================================
# CONFUSION MATRIX
# ============================================================

print("\n")
print("=" * 70)
print("CONFUSION MATRIX")
print("=" * 70)

print(
    cm
)


# ============================================================
# CLASSIFICATION REPORT
# ============================================================

print("\n")
print("=" * 70)
print("CLASSIFICATION REPORT")
print("=" * 70)

print(
    classification_report(
        y,
        predictions,
        target_names=[
            "Class 0",
            "Class 1"
        ],
        digits=4,
        zero_division=0
    )
)


# ============================================================
# SAVE RESULTS
# ============================================================

results_file = (
    "target_subject_results.txt"
)

with open(
    results_file,
    "w"
) as f:

    f.write(
        "Adversarial EEGNet Results\n"
    )

    f.write(
        "==========================\n\n"
    )

    f.write(
        f"Target subject: "
        f"{TARGET_SUBJECT}\n"
    )

    f.write(
        f"Source subjects: "
        f"{source_subjects}\n\n"
    )

    f.write(
        f"Accuracy: "
        f"{accuracy:.6f}\n"
    )

    f.write(
        f"Precision: "
        f"{precision:.6f}\n"
    )

    f.write(
        f"Recall: "
        f"{recall:.6f}\n"
    )

    f.write(
        f"F1-score: "
        f"{f1:.6f}\n"
    )

    f.write(
        f"Cohen Kappa: "
        f"{kappa:.6f}\n\n"
    )

    f.write(
        "Confusion Matrix:\n"
    )

    f.write(
        str(cm)
    )


print("\n")
print(
    "Results saved to:",
    results_file
)

print("\n")
print("=" * 70)
print("EVALUATION COMPLETE")
print("=" * 70)
