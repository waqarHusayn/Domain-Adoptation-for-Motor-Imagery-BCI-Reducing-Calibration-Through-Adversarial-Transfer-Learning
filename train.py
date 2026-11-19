import os
import random

import numpy as np
import torch
import torch.nn as nn

from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader

from sklearn.model_selection import train_test_split

from dataset import load_multiple_subjects
from model import AdversarialEEGNet


# ============================================================
# CONFIGURATION
# ============================================================

DATA_DIR = "BCICIV_2b_gdf"

# ------------------------------------------------------------
# IMPORTANT
#
# Subject 1 is the TARGET.
#
# Subjects 2-9 are SOURCE domains.
# ------------------------------------------------------------

TARGET_SUBJECT = 1

SOURCE_SUBJECTS = [
    2,
    3,
    4,
    5,
    6,
    7,
    8,
    9
]

# ------------------------------------------------------------

BATCH_SIZE = 32

EPOCHS = 50

LEARNING_RATE = 0.001

WEIGHT_DECAY = 1e-4

VALIDATION_SIZE = 0.20

RANDOM_SEED = 42

MODEL_PATH = "adversarial_eegnet.pth"


# ============================================================
# REPRODUCIBILITY
# ============================================================

def set_seed(seed):

    random.seed(seed)

    np.random.seed(seed)

    torch.manual_seed(seed)

    if torch.cuda.is_available():

        torch.cuda.manual_seed_all(seed)


set_seed(
    RANDOM_SEED
)


# ============================================================
# DEVICE
# ============================================================

device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

print("=" * 70)

print(
    "DEVICE:",
    device
)

if torch.cuda.is_available():

    print(
        "GPU:",
        torch.cuda.get_device_name(0)
    )

print("=" * 70)


# ============================================================
# LOAD SOURCE DATA
# ============================================================

print("\n")
print("=" * 70)
print("LOADING SOURCE SUBJECTS")
print("=" * 70)

X_source, y_source, domain_source = (
    load_multiple_subjects(
        SOURCE_SUBJECTS,
        DATA_DIR
    )
)

print("\nSource dataset:")

print(
    "X:",
    X_source.shape
)

print(
    "y:",
    y_source.shape
)

print(
    "domain:",
    domain_source.shape
)

print(
    "Number of domains:",
    len(SOURCE_SUBJECTS)
)


# ============================================================
# TRAIN / VALIDATION SPLIT
# ============================================================

indices = np.arange(
    len(y_source)
)

train_indices, val_indices = (
    train_test_split(
        indices,
        test_size=VALIDATION_SIZE,
        random_state=RANDOM_SEED,
        stratify=y_source
    )
)


X_train = X_source[
    train_indices
]

y_train = y_source[
    train_indices
]

domain_train = domain_source[
    train_indices
]


X_val = X_source[
    val_indices
]

y_val = y_source[
    val_indices
]

domain_val = domain_source[
    val_indices
]


print("\nTraining samples:", len(y_train))

print("Validation samples:", len(y_val))


# ============================================================
# CONVERT TO PYTORCH
# ============================================================

# Original:
#
# [samples, channels, time]
#
# EEGNet expects:
#
# [samples, 1, channels, time]

X_train = torch.tensor(
    X_train,
    dtype=torch.float32
).unsqueeze(1)

y_train = torch.tensor(
    y_train,
    dtype=torch.long
)

domain_train = torch.tensor(
    domain_train,
    dtype=torch.long
)


X_val = torch.tensor(
    X_val,
    dtype=torch.float32
).unsqueeze(1)

y_val = torch.tensor(
    y_val,
    dtype=torch.long
)

domain_val = torch.tensor(
    domain_val,
    dtype=torch.long
)


# ============================================================
# DATA LOADERS
# ============================================================

train_dataset = TensorDataset(
    X_train,
    y_train,
    domain_train
)

val_dataset = TensorDataset(
    X_val,
    y_val,
    domain_val
)


train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0
)


# ============================================================
# MODEL
# ============================================================

NUM_DOMAINS = len(
    SOURCE_SUBJECTS
)

model = AdversarialEEGNet(
    channels=3,
    samples=1000,
    num_classes=2,
    num_domains=NUM_DOMAINS,
    lambda_value=0.0
).to(device)


# ============================================================
# LOSS FUNCTIONS
# ============================================================

classification_loss = nn.CrossEntropyLoss()

domain_loss = nn.CrossEntropyLoss()


# ============================================================
# OPTIMIZER
# ============================================================

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=LEARNING_RATE,
    weight_decay=WEIGHT_DECAY
)


# ============================================================
# ACCURACY FUNCTION
# ============================================================

def accuracy(
    predictions,
    targets
):

    predicted_labels = (
        predictions.argmax(
            dim=1
        )
    )

    correct = (
        predicted_labels == targets
    ).sum().item()

    total = len(targets)

    return correct / total


# ============================================================
# TRAINING
# ============================================================

best_val_accuracy = 0.0


print("\n")
print("=" * 70)
print("STARTING ADVERSARIAL TRAINING")
print("=" * 70)


for epoch in range(1, EPOCHS + 1):

    # --------------------------------------------------------
    # DANN lambda schedule
    #
    # Starts close to 0 and gradually approaches 1.
    # --------------------------------------------------------

    progress = (
        epoch - 1
    ) / max(
        EPOCHS - 1,
        1
    )

    lambda_value = (
        2.0 / (
            1.0 +
            np.exp(
                -10.0 * progress
            )
        )
        - 1.0
    )

    model.set_lambda(
        lambda_value
    )

    # --------------------------------------------------------
    # TRAIN
    # --------------------------------------------------------

    model.train()

    total_class_loss = 0.0

    total_domain_loss = 0.0

    total_samples = 0

    total_class_correct = 0

    total_domain_correct = 0

    for (
        eeg,
        labels,
        domains
    ) in train_loader:

        eeg = eeg.to(device)

        labels = labels.to(device)

        domains = domains.to(device)

        optimizer.zero_grad()

        # ----------------------------------------------------
        # Forward
        # ----------------------------------------------------

        class_output, domain_output = (
            model(eeg)
        )

        # ----------------------------------------------------
        # Losses
        # ----------------------------------------------------

        loss_class = (
            classification_loss(
                class_output,
                labels
            )
        )

        loss_domain = (
            domain_loss(
                domain_output,
                domains
            )
        )

        # ----------------------------------------------------
        # Total loss
        #
        # GRL already reverses the domain gradient.
        # Therefore we ADD both losses here.
        # ----------------------------------------------------

        loss = (
            loss_class
            + loss_domain
        )

        # ----------------------------------------------------
        # Backpropagation
        # ----------------------------------------------------

        loss.backward()

        optimizer.step()

        # ----------------------------------------------------
        # Statistics
        # ----------------------------------------------------

        batch_size = eeg.size(0)

        total_samples += batch_size

        total_class_loss += (
            loss_class.item()
            * batch_size
        )

        total_domain_loss += (
            loss_domain.item()
            * batch_size
        )

        total_class_correct += (
            (
                class_output.argmax(1)
                == labels
            )
            .sum()
            .item()
        )

        total_domain_correct += (
            (
                domain_output.argmax(1)
                == domains
            )
            .sum()
            .item()
        )

    train_class_loss = (
        total_class_loss
        / total_samples
    )

    train_domain_loss = (
        total_domain_loss
        / total_samples
    )

    train_class_acc = (
        total_class_correct
        / total_samples
    )

    train_domain_acc = (
        total_domain_correct
        / total_samples
    )

    # --------------------------------------------------------
    # VALIDATION
    # --------------------------------------------------------

    model.eval()

    val_correct = 0

    val_total = 0

    val_loss = 0.0

    with torch.no_grad():

        for (
            eeg,
            labels,
            domains
        ) in val_loader:

            eeg = eeg.to(device)

            labels = labels.to(device)

            domains = domains.to(device)

            class_output, domain_output = (
                model(eeg)
            )

            loss = (
                classification_loss(
                    class_output,
                    labels
                )
            )

            val_loss += (
                loss.item()
                * eeg.size(0)
            )

            val_correct += (
                (
                    class_output.argmax(1)
                    == labels
                )
                .sum()
                .item()
            )

            val_total += (
                eeg.size(0)
            )

    val_loss /= val_total

    val_accuracy = (
        val_correct
        / val_total
    )

    # --------------------------------------------------------
    # SAVE BEST MODEL
    # --------------------------------------------------------

    if val_accuracy > best_val_accuracy:

        best_val_accuracy = (
            val_accuracy
        )

        torch.save(
            {
                "model_state_dict":
                    model.state_dict(),

                "source_subjects":
                    SOURCE_SUBJECTS,

                "target_subject":
                    TARGET_SUBJECT,

                "best_val_accuracy":
                    best_val_accuracy,

                "epoch":
                    epoch
            },
            MODEL_PATH
        )

        saved = " <-- SAVED"

    else:

        saved = ""

    # --------------------------------------------------------
    # PRINT
    # --------------------------------------------------------

    print(
        f"Epoch {epoch:03d}/{EPOCHS} | "
        f"lambda={lambda_value:.3f} | "
        f"class_loss={train_class_loss:.4f} | "
        f"domain_loss={train_domain_loss:.4f} | "
        f"class_acc={train_class_acc:.4f} | "
        f"domain_acc={train_domain_acc:.4f} | "
        f"val_acc={val_accuracy:.4f}"
        f"{saved}"
    )


# ============================================================
# TRAINING COMPLETE
# ============================================================

print("\n")
print("=" * 70)
print("TRAINING COMPLETE")
print("=" * 70)

print(
    "Best validation accuracy:",
    f"{best_val_accuracy:.4f}"
)

print(
    "Saved model:",
    MODEL_PATH
)
