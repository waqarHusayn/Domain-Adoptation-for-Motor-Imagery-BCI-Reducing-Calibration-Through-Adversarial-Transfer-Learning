# ============================================================
# BCI COMPETITION IV 2B
# EEGNet + Optional Domain Adversarial Training
#
# COMPLETE TRAINING + 50-TRIAL HOLDOUT TESTING + LOSO
#
# Requirements:
#   numpy
#   scipy
#   pandas
#   scikit-learn
#   matplotlib
#   mne
#   torch
#
# Dataset:
#   BCI Competition IV Dataset 2b
#
# Expected folder:
#   C:\Users\Student.DESKTOP-RUSD6ON.001\Desktop\FYP\BCICIV_2b_gdf
#
# ============================================================

import os
import random
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import mne
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import (
    Dataset,
    DataLoader,
)

from sklearn.model_selection import (
    train_test_split,
    StratifiedShuffleSplit,
)

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    cohen_kappa_score,
    confusion_matrix,
    classification_report,
)

warnings.filterwarnings("ignore")

# ============================================================
# CONFIGURATION
# ============================================================

SEED = 42

BASE_DIR = Path(
    r"C:\Users\Student.DESKTOP-RUSD6ON.001\Desktop\FYP"
)

DATASET_DIR = BASE_DIR / "BCICIV_2b_gdf"

RESULTS_DIR = BASE_DIR / "results"
MODELS_DIR = BASE_DIR / "models"

RESULTS_DIR.mkdir(exist_ok=True)
MODELS_DIR.mkdir(exist_ok=True)

MODEL_PATH = MODELS_DIR / "eegnet_50trial_best.pth"

# ------------------------------------------------------------
# Dataset
# ------------------------------------------------------------

SUBJECTS = list(range(1, 10))

EEG_CHANNELS = [
    "EEG:C3",
    "EEG:Cz",
    "EEG:C4",
]

LOW_FREQ = 4.0
HIGH_FREQ = 40.0

# BCI IV 2b sampling frequency is 250 Hz.
SFREQ = 250

# Start after cue.
# 4 seconds = 1000 samples.
TRIAL_START_SEC = 0.0
TRIAL_DURATION_SEC = 4.0

TRIAL_SAMPLES = int(
    TRIAL_DURATION_SEC * SFREQ
)

# ------------------------------------------------------------
# Test configuration
# ------------------------------------------------------------

TOTAL_TEST_TRIALS = 50

# Test is selected before model training.
# This prevents test leakage.
TEST_FRACTION = 0.125

# Validation fraction from remaining training data.
VAL_FRACTION = 0.15

# ------------------------------------------------------------
# Training
# ------------------------------------------------------------

BATCH_SIZE = 32

NUM_EPOCHS = 100

LEARNING_RATE = 0.001

WEIGHT_DECAY = 1e-4

PATIENCE = 10

MIN_DELTA = 0.0005

# Domain adversarial weight.
# Set to 0.0 if you want ordinary EEGNet.
DOMAIN_WEIGHT = 0.10

NUM_CLASSES = 2

# ------------------------------------------------------------
# Device
# ------------------------------------------------------------

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

# ============================================================
# REPRODUCIBILITY
# ============================================================

def set_seed(seed=SEED):

    random.seed(seed)

    np.random.seed(seed)

    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


set_seed()

print("=" * 70)
print("BCI IV 2B EEG TRAINING + 50-TRIAL TESTING")
print("=" * 70)

print(f"Device : {DEVICE}")
print(f"Dataset: {DATASET_DIR}")
print()


# ============================================================
# DATASET CLASS
# ============================================================

class EEGDataset(Dataset):

    def __init__(
        self,
        X,
        y,
        subject_ids=None
    ):

        self.X = torch.tensor(
            X,
            dtype=torch.float32
        )

        self.y = torch.tensor(
            y,
            dtype=torch.long
        )

        if subject_ids is None:

            subject_ids = np.zeros(
                len(y),
                dtype=np.int64
            )

        self.subject_ids = torch.tensor(
            subject_ids,
            dtype=torch.long
        )

    def __len__(self):

        return len(self.y)

    def __getitem__(self, index):

        return (
            self.X[index],
            self.y[index],
            self.subject_ids[index]
        )


# ============================================================
# GRADIENT REVERSAL
# ============================================================

class GradientReversalFunction(
    torch.autograd.Function
):

    @staticmethod
    def forward(
        ctx,
        x,
        lambda_value
    ):

        ctx.lambda_value = lambda_value

        return x.view_as(x)

    @staticmethod
    def backward(
        ctx,
        grad_output
    ):

        return (
            -ctx.lambda_value * grad_output,
            None
        )


class GradientReversal(nn.Module):

    def __init__(self):

        super().__init__()

    def forward(
        self,
        x,
        lambda_value
    ):

        return GradientReversalFunction.apply(
            x,
            lambda_value
        )


# ============================================================
# EEGNET
# ============================================================

class EEGNetFeatureExtractor(nn.Module):

    def __init__(
        self,
        channels=3,
        samples=1000
    ):

        super().__init__()

        self.first_conv = nn.Sequential(

            nn.Conv2d(
                1,
                8,
                kernel_size=(1, 64),
                padding=(0, 32),
                bias=False
            ),

            nn.BatchNorm2d(8)
        )

        self.depthwise_conv = nn.Sequential(

            nn.Conv2d(
                8,
                16,
                kernel_size=(channels, 1),
                groups=8,
                bias=False
            ),

            nn.BatchNorm2d(16),

            nn.ELU(),

            nn.AvgPool2d(
                kernel_size=(1, 4)
            ),

            nn.Dropout(0.25)
        )

        self.separable_conv = nn.Sequential(

            nn.Conv2d(
                16,
                16,
                kernel_size=(1, 16),
                padding=(0, 8),
                groups=16,
                bias=False
            ),

            nn.Conv2d(
                16,
                16,
                kernel_size=(1, 1),
                bias=False
            ),

            nn.BatchNorm2d(16),

            nn.ELU(),

            nn.AvgPool2d(
                kernel_size=(1, 8)
            ),

            nn.Dropout(0.25)
        )

        # Determine feature dimension automatically.
        with torch.no_grad():

            dummy = torch.zeros(
                1,
                1,
                channels,
                samples
            )

            output = self.forward_features(
                dummy
            )

            self.feature_dim = output.shape[1]

    def forward_features(self, x):

        x = self.first_conv(x)

        x = self.depthwise_conv(x)

        x = self.separable_conv(x)

        x = torch.flatten(
            x,
            start_dim=1
        )

        return x

    def forward(self, x):

        return self.forward_features(x)


# ============================================================
# COMPLETE MODEL
# ============================================================

class EEGNetDomainAdversarial(
    nn.Module
):

    def __init__(
        self,
        channels=3,
        samples=1000,
        num_classes=2,
        num_domains=9
    ):

        super().__init__()

        self.feature_extractor = (
            EEGNetFeatureExtractor(
                channels=channels,
                samples=samples
            )
        )

        feature_dim = (
            self.feature_extractor.feature_dim
        )

        self.classifier = nn.Sequential(

            nn.Linear(
                feature_dim,
                64
            ),

            nn.ELU(),

            nn.Dropout(0.5),

            nn.Linear(
                64,
                num_classes
            )
        )

        self.domain_classifier = nn.Sequential(

            nn.Linear(
                feature_dim,
                64
            ),

            nn.ELU(),

            nn.Dropout(0.5),

            nn.Linear(
                64,
                num_domains
            )
        )

        self.gradient_reversal = (
            GradientReversal()
        )

    def forward(
        self,
        x,
        domain_lambda=0.0
    ):

        features = (
            self.feature_extractor(x)
        )

        class_output = (
            self.classifier(features)
        )

        if domain_lambda > 0:

            reversed_features = (
                self.gradient_reversal(
                    features,
                    domain_lambda
                )
            )

            domain_output = (
                self.domain_classifier(
                    reversed_features
                )
            )

        else:

            domain_output = None

        return (
            class_output,
            domain_output,
            features
        )


# ============================================================
# GDF LOADING
# ============================================================

def find_eeg_channels(raw):

    available = raw.ch_names

    selected = []

    for desired in EEG_CHANNELS:

        if desired in available:

            selected.append(desired)

            continue

        desired_clean = (
            desired
            .replace("EEG:", "")
            .upper()
        )

        for channel in available:

            channel_clean = (
                channel
                .replace("EEG:", "")
                .upper()
            )

            if channel_clean == desired_clean:

                selected.append(channel)

                break

    if len(selected) != 3:

        raise RuntimeError(
            f"Could not identify C3/Cz/C4.\n"
            f"Available channels: {available}"
        )

    return selected


def get_motor_imagery_event_ids(
    event_dict
):

    event_769 = None
    event_770 = None

    for key, value in event_dict.items():

        key_str = str(key)

        if key_str == "769":
            event_769 = value

        elif key_str == "770":
            event_770 = value

    if (
        event_769 is None
        or event_770 is None
    ):

        return None, None

    return (
        event_769,
        event_770
    )


def load_gdf_file(
    filepath,
    verbose=True
):

    filepath = Path(filepath)

    if verbose:

        print("\n" + "=" * 70)
        print(f"Loading: {filepath}")
        print("=" * 70)

    raw = mne.io.read_raw_gdf(
        str(filepath),
        preload=True,
        verbose=False
    )

    if verbose:

        print("Channels:")
        print(raw.ch_names)

    selected_channels = (
        find_eeg_channels(raw)
    )

    if verbose:

        print("\nSelected EEG channels:")
        print(selected_channels)

    raw.pick(selected_channels)

    if verbose:

        print(
            f"\nRaw EEG shape: "
            f"{raw.get_data().shape}"
        )

        print(
            f"\nApplying "
            f"{LOW_FREQ}-{HIGH_FREQ} Hz "
            f"band-pass filter..."
        )

    raw.filter(
        l_freq=LOW_FREQ,
        h_freq=HIGH_FREQ,
        verbose=False
    )

    data = raw.get_data()

    if verbose:

        print("\nReading annotations...")

    events, event_dict = (
        mne.events_from_annotations(
            raw,
            verbose=False
        )
    )

    if verbose:

        print("\nEvent dictionary:")
        print(event_dict)

    event_769, event_770 = (
        get_motor_imagery_event_ids(
            event_dict
        )
    )

    if (
        event_769 is None
        or event_770 is None
    ):

        raise RuntimeError(
            f"Could not find motor imagery "
            f"events 769/770 in {filepath}"
        )

    if verbose:

        print(
            f"Motor imagery IDs: "
            f"769 -> {event_769}, "
            f"770 -> {event_770}"
        )

    X_trials = []
    y_trials = []

    samples_per_trial = (
        TRIAL_SAMPLES
    )

    for event in events:

        sample_index = int(
            event[0]
        )

        event_id = int(
            event[2]
        )

        if event_id == event_769:

            label = 0

        elif event_id == event_770:

            label = 1

        else:

            continue

        start = (
            sample_index
            + int(
                TRIAL_START_SEC
                * SFREQ
            )
        )

        end = (
            start
            + samples_per_trial
        )

        if start < 0:
            continue

        if end > data.shape[1]:
            continue

        trial = data[
            :,
            start:end
        ]

        if trial.shape[1] != samples_per_trial:
            continue

        X_trials.append(trial)

        y_trials.append(label)

    if len(X_trials) == 0:

        raise RuntimeError(
            f"No motor imagery trials "
            f"found in {filepath}"
        )

    X = np.asarray(
        X_trials,
        dtype=np.float32
    )

    y = np.asarray(
        y_trials,
        dtype=np.int64
    )

    if verbose:

        print(
            f"\nTrials: {X.shape}"
        )

        print(
            f"Labels: {y.shape}"
        )

        print(
            f"Class distribution: "
            f"{np.bincount(y)}"
        )

    return X, y


# ============================================================
# LOAD SUBJECT
# ============================================================

def load_subject(
    subject,
    verbose=True
):

    subject_dir = Path(
        DATASET_DIR
    )

    training_files = [

        subject_dir
        / f"B{subject:02d}01T.gdf",

        subject_dir
        / f"B{subject:02d}02T.gdf",

        subject_dir
        / f"B{subject:02d}03T.gdf",
    ]

    if verbose:

        print(
            "\n"
            + "#" * 70
        )

        print(
            f"LOADING SUBJECT "
            f"{subject:02d}"
        )

        print(
            "#" * 70
        )

    all_X = []
    all_y = []

    for filepath in training_files:

        if not filepath.exists():

            print(
                f"WARNING: File not found: "
                f"{filepath}"
            )

            continue

        X, y = load_gdf_file(
            filepath,
            verbose=verbose
        )

        all_X.append(X)

        all_y.append(y)

    if len(all_X) == 0:

        raise RuntimeError(
            f"No labeled training files "
            f"found for subject "
            f"{subject:02d}"
        )

    X_subject = np.concatenate(
        all_X,
        axis=0
    )

    y_subject = np.concatenate(
        all_y,
        axis=0
    )

    if verbose:

        print("\nSubject summary:")

        print(
            f"X: {X_subject.shape}"
        )

        print(
            f"y: {y_subject.shape}"
        )

        print(
            f"Class distribution: "
            f"{np.bincount(y_subject)}"
        )

    return X_subject, y_subject


# ============================================================
# LOAD ALL SUBJECTS
# ============================================================

def load_all_subjects():

    all_X = []
    all_y = []
    all_subject_ids = []

    for subject in SUBJECTS:

        X, y = load_subject(
            subject
        )

        all_X.append(X)

        all_y.append(y)

        subject_ids = np.full(
            len(y),
            subject,
            dtype=np.int64
        )

        all_subject_ids.append(
            subject_ids
        )

    X = np.concatenate(
        all_X,
        axis=0
    )

    y = np.concatenate(
        all_y,
        axis=0
    )

    subject_ids = np.concatenate(
        all_subject_ids,
        axis=0
    )

    print(
        "\n"
        + "=" * 70
    )

    print(
        "ALL LABELED TRAINING DATA"
    )

    print(
        "=" * 70
    )

    print(
        f"X shape      : {X.shape}"
    )

    print(
        f"y shape      : {y.shape}"
    )

    print(
        f"Subject IDs  : "
        f"{np.unique(subject_ids)}"
    )

    print(
        f"Class counts : "
        f"{np.bincount(y)}"
    )

    return (
        X,
        y,
        subject_ids
    )


# ============================================================
# NORMALIZATION
# ============================================================

def normalize_data(
    X_train,
    X_val,
    X_test
):

    # Calculate statistics ONLY from training data.
    #
    # This prevents validation/test leakage.

    mean = X_train.mean()

    std = X_train.std()

    if std < 1e-8:

        std = 1.0

    X_train = (
        X_train - mean
    ) / std

    X_val = (
        X_val - mean
    ) / std

    X_test = (
        X_test - mean
    ) / std

    return (
        X_train.astype(np.float32),
        X_val.astype(np.float32),
        X_test.astype(np.float32),
        float(mean),
        float(std)
    )


# ============================================================
# CREATE EXACT 50-TRIAL HOLDOUT
# ============================================================

def create_50_trial_test_set(
    X,
    y,
    subject_ids
):

    rng = np.random.RandomState(
        SEED
    )

    candidate_indices = (
        np.arange(len(y))
    )

    # Stratify by class so the 50 trials
    # contain 25 samples per class.

    test_indices, remaining_indices = (
        train_test_split(
            candidate_indices,
            test_size=(
                len(candidate_indices)
                - TOTAL_TEST_TRIALS
            ),
            stratify=y,
            random_state=SEED
        )
    )

    # train_test_split returns the requested
    # smaller partition first.
    #
    # Verify exactly 50.

    if len(test_indices) != TOTAL_TEST_TRIALS:

        raise RuntimeError(
            f"Expected "
            f"{TOTAL_TEST_TRIALS} "
            f"test trials, got "
            f"{len(test_indices)}"
        )

    X_test = X[test_indices]

    y_test = y[test_indices]

    subject_test = (
        subject_ids[test_indices]
    )

    X_remaining = X[
        remaining_indices
    ]

    y_remaining = y[
        remaining_indices
    ]

    subject_remaining = (
        subject_ids[
            remaining_indices
        ]
    )

    print(
        "\n"
        + "=" * 70
    )

    print(
        "50-TRIAL HOLDOUT TEST SET"
    )

    print(
        "=" * 70
    )

    print(
        f"Test trials: "
        f"{len(X_test)}"
    )

    print(
        f"Class distribution: "
        f"{np.bincount(y_test)}"
    )

    print(
        "Subject distribution:"
    )

    for subject in SUBJECTS:

        count = np.sum(
            subject_test == subject
        )

        if count > 0:

            print(
                f"  Subject {subject:02d}: "
                f"{count}"
            )

    return (
        X_remaining,
        y_remaining,
        subject_remaining,
        X_test,
        y_test,
        subject_test
    )


# ============================================================
# TRAIN / VALIDATION SPLIT
# ============================================================

def create_train_val_split(
    X,
    y,
    subject_ids
):

    indices = np.arange(
        len(y)
    )

    train_indices, val_indices = (
        train_test_split(
            indices,
            test_size=VAL_FRACTION,
            stratify=y,
            random_state=SEED
        )
    )

    X_train = X[
        train_indices
    ]

    y_train = y[
        train_indices
    ]

    subject_train = subject_ids[
        train_indices
    ]

    X_val = X[
        val_indices
    ]

    y_val = y[
        val_indices
    ]

    subject_val = subject_ids[
        val_indices
    ]

    return (
        X_train,
        y_train,
        subject_train,
        X_val,
        y_val,
        subject_val
    )


# ============================================================
# TRAIN ONE EPOCH
# ============================================================

def train_one_epoch(
    model,
    loader,
    optimizer,
    class_criterion,
    domain_criterion,
    epoch,
    total_epochs
):

    model.train()

    total_class_loss = 0.0

    total_domain_loss = 0.0

    correct = 0

    total = 0

    batches = 0

    # Gradually increase domain adaptation.
    progress = (
        epoch
        / max(
            1,
            total_epochs - 1
        )
    )

    domain_lambda = (
        DOMAIN_WEIGHT
        * progress
    )

    for X_batch, y_batch, subject_batch in loader:

        X_batch = X_batch.to(
            DEVICE
        )

        y_batch = y_batch.to(
            DEVICE
        )

        subject_batch = (
            subject_batch.to(
                DEVICE
            )
        )

        # EEGNet expects:
        #
        # batch x 1 x channels x samples

        X_batch = X_batch.unsqueeze(1)

        optimizer.zero_grad()

        class_output, domain_output, _ = (
            model(
                X_batch,
                domain_lambda=domain_lambda
            )
        )

        class_loss = (
            class_criterion(
                class_output,
                y_batch
            )
        )

        if (
            DOMAIN_WEIGHT > 0
            and domain_output is not None
        ):

            # Convert subject IDs to 0-8.
            domain_labels = (
                subject_batch - 1
            )

            domain_loss = (
                domain_criterion(
                    domain_output,
                    domain_labels
                )
            )

        else:

            domain_loss = torch.tensor(
                0.0,
                device=DEVICE
            )

        loss = (
            class_loss
            + DOMAIN_WEIGHT
            * domain_loss
        )

        loss.backward()

        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            max_norm=5.0
        )

        optimizer.step()

        total_class_loss += (
            class_loss.item()
        )

        total_domain_loss += (
            domain_loss.item()
        )

        predictions = (
            torch.argmax(
                class_output,
                dim=1
            )
        )

        correct += (
            predictions == y_batch
        ).sum().item()

        total += (
            y_batch.size(0)
        )

        batches += 1

    return (
        total_class_loss / max(
            1,
            batches
        ),
        total_domain_loss / max(
            1,
            batches
        ),
        correct / max(
            1,
            total
        )
    )


# ============================================================
# VALIDATION
# ============================================================

@torch.no_grad()
def evaluate_loader(
    model,
    loader
):

    model.eval()

    predictions = []

    targets = []

    for X_batch, y_batch, _ in loader:

        X_batch = X_batch.to(
            DEVICE
        )

        X_batch = X_batch.unsqueeze(1)

        output, _, _ = model(
            X_batch,
            domain_lambda=0.0
        )

        pred = torch.argmax(
            output,
            dim=1
        )

        predictions.extend(
            pred.cpu().numpy()
        )

        targets.extend(
            y_batch.numpy()
        )

    predictions = np.asarray(
        predictions
    )

    targets = np.asarray(
        targets
    )

    accuracy = accuracy_score(
        targets,
        predictions
    )

    return (
        accuracy,
        targets,
        predictions
    )


# ============================================================
# METRICS
# ============================================================

def calculate_metrics(
    y_true,
    y_pred
):

    cm = confusion_matrix(
        y_true,
        y_pred,
        labels=[0, 1]
    )

    return {

        "accuracy":
            accuracy_score(
                y_true,
                y_pred
            ),

        "precision":
            precision_score(
                y_true,
                y_pred,
                zero_division=0
            ),

        "recall":
            recall_score(
                y_true,
                y_pred,
                zero_division=0
            ),

        "f1":
            f1_score(
                y_true,
                y_pred,
                zero_division=0
            ),

        "kappa":
            cohen_kappa_score(
                y_true,
                y_pred
            ),

        "confusion_matrix":
            cm
    }


# ============================================================
# PREDICT
# ============================================================

@torch.no_grad()
def predict(
    model,
    X
):

    model.eval()

    dataset = EEGDataset(
        X,
        np.zeros(
            len(X),
            dtype=np.int64
        )
    )

    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False
    )

    predictions = []

    probabilities = []

    for X_batch, _, _ in loader:

        X_batch = X_batch.to(
            DEVICE
        )

        X_batch = X_batch.unsqueeze(1)

        logits, _, _ = model(
            X_batch,
            domain_lambda=0.0
        )

        probs = torch.softmax(
            logits,
            dim=1
        )

        pred = torch.argmax(
            probs,
            dim=1
        )

        predictions.extend(
            pred.cpu().numpy()
        )

        probabilities.extend(
            probs.cpu().numpy()
        )

    return (
        np.asarray(predictions),
        np.asarray(probabilities)
    )


# ============================================================
# SAVE CONFUSION MATRIX
# ============================================================

def save_confusion_matrix(
    cm,
    filename,
    title
):

    fig, ax = plt.subplots(
        figsize=(6, 5)
    )

    im = ax.imshow(
        cm,
        interpolation="nearest",
        cmap="Blues"
    )

    ax.figure.colorbar(im, ax=ax)

    ax.set(
        xticks=[0, 1],
        yticks=[0, 1],
        xticklabels=[
            "Class 0",
            "Class 1"
        ],
        yticklabels=[
            "Class 0",
            "Class 1"
        ],
        ylabel="True Label",
        xlabel="Predicted Label",
        title=title
    )

    threshold = (
        cm.max() / 2.0
    )

    for i in range(2):

        for j in range(2):

            ax.text(
                j,
                i,
                str(cm[i, j]),
                ha="center",
                va="center",
                color=(
                    "white"
                    if cm[i, j] > threshold
                    else "black"
                ),
                fontsize=14
            )

    fig.tight_layout()

    plt.savefig(
        filename,
        dpi=200,
        bbox_inches="tight"
    )

    plt.close()


# ============================================================
# TRAINING
# ============================================================

def train_model(
    X_train,
    y_train,
    subject_train,
    X_val,
    y_val,
    subject_val
):

    train_dataset = EEGDataset(
        X_train,
        y_train,
        subject_train
    )

    val_dataset = EEGDataset(
        X_val,
        y_val,
        subject_val
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        drop_last=False
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False
    )

    model = EEGNetDomainAdversarial(
        channels=3,
        samples=TRIAL_SAMPLES,
        num_classes=2,
        num_domains=len(SUBJECTS)
    ).to(DEVICE)

    class_criterion = nn.CrossEntropyLoss()

    domain_criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=0.5,
        patience=5
    )

    print(
        "\nTrainable parameters:",
        sum(
            p.numel()
            for p in model.parameters()
            if p.requires_grad
        )
    )

    print(
        "\n"
        + "=" * 70
    )

    print("TRAINING")

    print(
        "=" * 70
    )

    print(
        f"Epochs       : {NUM_EPOCHS}"
    )

    print(
        f"Batch size   : {BATCH_SIZE}"
    )

    print(
        f"Learning rate: {LEARNING_RATE}"
    )

    print(
        f"Domain weight: {DOMAIN_WEIGHT}"
    )

    best_val_accuracy = 0.0

    best_epoch = 0

    patience_counter = 0

    history = []

    for epoch in range(
        NUM_EPOCHS
    ):

        class_loss, domain_loss, train_acc = (
            train_one_epoch(
                model,
                train_loader,
                optimizer,
                class_criterion,
                domain_criterion,
                epoch,
                NUM_EPOCHS
            )
        )

        val_accuracy, _, _ = (
            evaluate_loader(
                model,
                val_loader
            )
        )

        scheduler.step(
            val_accuracy
        )

        current_lr = (
            optimizer.param_groups[0]["lr"]
        )

        print(
            f"Epoch "
            f"{epoch + 1:03d}/"
            f"{NUM_EPOCHS} | "
            f"Class Loss: "
            f"{class_loss:.4f} | "
            f"Domain Loss: "
            f"{domain_loss:.4f} | "
            f"Train Acc: "
            f"{train_acc * 100:.2f}% | "
            f"Val Acc: "
            f"{val_accuracy * 100:.2f}% | "
            f"LR: "
            f"{current_lr:.6f}"
        )

        history.append({

            "epoch":
                epoch + 1,

            "class_loss":
                class_loss,

            "domain_loss":
                domain_loss,

            "train_accuracy":
                train_acc,

            "val_accuracy":
                val_accuracy,

            "learning_rate":
                current_lr
        })

        if (
            val_accuracy
            >
            best_val_accuracy
            + MIN_DELTA
        ):

            best_val_accuracy = (
                val_accuracy
            )

            best_epoch = (
                epoch + 1
            )

            patience_counter = 0

            # IMPORTANT:
            # Only tensors and ordinary Python
            # numbers are saved.
            #
            # This works with:
            # torch.load(..., weights_only=True)

            torch.save(
                {
                    "model_state_dict":
                        model.state_dict(),

                    "best_val_acc":
                        float(
                            best_val_accuracy
                        ),

                    "epoch":
                        int(
                            best_epoch
                        )
                },
                MODEL_PATH
            )

            print(
                f"  -> Best model saved: "
                f"{best_val_accuracy * 100:.2f}%"
            )

        else:

            patience_counter += 1

        if (
            patience_counter
            >= PATIENCE
        ):

            print(
                "\nEarly stopping triggered."
            )

            break

    history_df = pd.DataFrame(
        history
    )

    history_df.to_csv(
        RESULTS_DIR
        / "training_history.csv",
        index=False
    )

    print(
        "\n"
        + "=" * 70
    )

    print(
        "LOADING BEST MODEL"
    )

    print(
        "=" * 70
    )

    if not MODEL_PATH.exists():

        raise FileNotFoundError(
            f"Model checkpoint not found: "
            f"{MODEL_PATH}"
        )

    checkpoint = torch.load(
        MODEL_PATH,
        map_location=DEVICE,
        weights_only=True
    )

    model.load_state_dict(
        checkpoint[
            "model_state_dict"
        ]
    )

    best_val_accuracy = float(
        checkpoint[
            "best_val_acc"
        ]
    )

    best_epoch = int(
        checkpoint[
            "epoch"
        ]
    )

    print(
        f"Best validation accuracy: "
        f"{best_val_accuracy * 100:.2f}%"
    )

    print(
        f"Best epoch: "
        f"{best_epoch}"
    )

    print(
        f"Model loaded from: "
        f"{MODEL_PATH}"
    )

    return (
        model,
        history_df,
        best_val_accuracy,
        best_epoch
    )


# ============================================================
# 50-TRIAL FINAL TEST
# ============================================================

def evaluate_50_trial_test(
    model,
    X_test,
    y_test,
    subject_test
):

    print(
        "\n"
        + "=" * 70
    )

    print(
        "FINAL 50-TRIAL TEST"
    )

    print(
        "=" * 70
    )

    print(
        "These trials were NOT used "
        "for training or validation."
    )

    predictions, probabilities = (
        predict(
            model,
            X_test
        )
    )

    metrics = calculate_metrics(
        y_test,
        predictions
    )

    print(
        f"\nNumber of test trials: "
        f"{len(y_test)}"
    )

    print(
        f"Test class distribution: "
        f"{np.bincount(y_test)}"
    )

    print(
        "\n"
        + "-" * 70
    )

    print(
        "50-TRIAL RESULTS"
    )

    print(
        "-" * 70
    )

    print(
        f"Accuracy : "
        f"{metrics['accuracy'] * 100:.2f}%"
    )

    print(
        f"Precision: "
        f"{metrics['precision'] * 100:.2f}%"
    )

    print(
        f"Recall   : "
        f"{metrics['recall'] * 100:.2f}%"
    )

    print(
        f"F1       : "
        f"{metrics['f1'] * 100:.2f}%"
    )

    print(
        f"Kappa    : "
        f"{metrics['kappa']:.4f}"
    )

    print(
        "\nConfusion matrix:"
    )

    print(
        metrics[
            "confusion_matrix"
        ]
    )

    print(
        "\nClassification report:"
    )

    print(
        classification_report(
            y_test,
            predictions,
            target_names=[
                "Class 0",
                "Class 1"
            ],
            zero_division=0
        )
    )

    # --------------------------------------------------------
    # Save predictions
    # --------------------------------------------------------

    test_results = pd.DataFrame({

        "trial":
            np.arange(
                1,
                len(y_test) + 1
            ),

        "subject":
            subject_test,

        "true_label":
            y_test,

        "predicted_label":
            predictions,

        "prob_class_0":
            probabilities[:, 0],

        "prob_class_1":
            probabilities[:, 1],

        "correct":
            (
                y_test
                == predictions
            )
    })

    test_results.to_csv(
        RESULTS_DIR
        / "50_trial_predictions.csv",
        index=False
    )

    # --------------------------------------------------------
    # Save metrics
    # --------------------------------------------------------

    metrics_df = pd.DataFrame([{

        "num_test_trials":
            len(y_test),

        "accuracy":
            metrics["accuracy"],

        "precision":
            metrics["precision"],

        "recall":
            metrics["recall"],

        "f1":
            metrics["f1"],

        "kappa":
            metrics["kappa"]

    }])

    metrics_df.to_csv(
        RESULTS_DIR
        / "50_trial_metrics.csv",
        index=False
    )

    save_confusion_matrix(
        metrics[
            "confusion_matrix"
        ],
        RESULTS_DIR
        / "50_trial_confusion_matrix.png",
        "50-Trial Test Confusion Matrix"
    )

    return (
        metrics,
        predictions,
        probabilities
    )


# ============================================================
# PER-SUBJECT EVALUATION
# ============================================================

def evaluate_each_subject(
    model,
    X,
    y,
    subject_ids
):

    print(
        "\n"
        + "=" * 70
    )

    print(
        "PER-SUBJECT EVALUATION"
    )

    print(
        "=" * 70
    )

    results = []

    for subject in SUBJECTS:

        mask = (
            subject_ids
            == subject
        )

        if np.sum(mask) == 0:
            continue

        X_subject = X[mask]

        y_subject = y[mask]

        predictions, _ = predict(
            model,
            X_subject
        )

        metrics = calculate_metrics(
            y_subject,
            predictions
        )

        results.append({

            "subject":
                subject,

            "num_trials":
                len(y_subject),

            "accuracy":
                metrics["accuracy"],

            "precision":
                metrics["precision"],

            "recall":
                metrics["recall"],

            "f1":
                metrics["f1"],

            "kappa":
                metrics["kappa"]
        })

        print(
            f"\nSubject {subject:02d}"
        )

        print(
            f"Trials   : "
            f"{len(y_subject)}"
        )

        print(
            f"Accuracy : "
            f"{metrics['accuracy'] * 100:.2f}%"
        )

        print(
            f"Precision: "
            f"{metrics['precision'] * 100:.2f}%"
        )

        print(
            f"Recall   : "
            f"{metrics['recall'] * 100:.2f}%"
        )

        print(
            f"F1       : "
            f"{metrics['f1'] * 100:.2f}%"
        )

        print(
            f"Kappa    : "
            f"{metrics['kappa']:.4f}"
        )

        save_confusion_matrix(
            metrics[
                "confusion_matrix"
            ],
            RESULTS_DIR
            / f"subject_{subject:02d}_confusion_matrix.png",
            f"Subject {subject:02d}"
        )

    results_df = pd.DataFrame(
        results
    )

    results_df.to_csv(
        RESULTS_DIR
        / "per_subject_results.csv",
        index=False
    )

    if len(results_df) > 0:

        summary = pd.DataFrame([{

            "accuracy_mean":
                results_df[
                    "accuracy"
                ].mean(),

            "accuracy_std":
                results_df[
                    "accuracy"
                ].std(),

            "precision_mean":
                results_df[
                    "precision"
                ].mean(),

            "precision_std":
                results_df[
                    "precision"
                ].std(),

            "recall_mean":
                results_df[
                    "recall"
                ].mean(),

            "recall_std":
                results_df[
                    "recall"
                ].std(),

            "f1_mean":
                results_df[
                    "f1"
                ].mean(),

            "f1_std":
                results_df[
                    "f1"
                ].std(),

            "kappa_mean":
                results_df[
                    "kappa"
                ].mean(),

            "kappa_std":
                results_df[
                    "kappa"
                ].std()
        }])

        summary.to_csv(
            RESULTS_DIR
            / "per_subject_summary.csv",
            index=False
        )

        print(
            "\n"
            + "=" * 70
        )

        print(
            "PER-SUBJECT MEAN ± STD"
        )

        print(
            "=" * 70
        )

        print(
            f"Accuracy : "
            f"{summary.iloc[0]['accuracy_mean'] * 100:.2f}"
            f" ± "
            f"{summary.iloc[0]['accuracy_std'] * 100:.2f}%"
        )

        print(
            f"Precision: "
            f"{summary.iloc[0]['precision_mean'] * 100:.2f}"
            f" ± "
            f"{summary.iloc[0]['precision_std'] * 100:.2f}%"
        )

        print(
            f"Recall   : "
            f"{summary.iloc[0]['recall_mean'] * 100:.2f}"
            f" ± "
            f"{summary.iloc[0]['recall_std'] * 100:.2f}%"
        )

        print(
            f"F1       : "
            f"{summary.iloc[0]['f1_mean'] * 100:.2f}"
            f" ± "
            f"{summary.iloc[0]['f1_std'] * 100:.2f}%"
        )

        print(
            f"Kappa    : "
            f"{summary.iloc[0]['kappa_mean']:.4f}"
            f" ± "
            f"{summary.iloc[0]['kappa_std']:.4f}"
        )

    return results_df


# ============================================================
# PLOT TRAINING CURVES
# ============================================================

def plot_training_history(
    history
):

    if len(history) == 0:
        return

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(12, 4)
    )

    axes[0].plot(
        history["epoch"],
        history["train_accuracy"]
        * 100,
        label="Train"
    )

    axes[0].plot(
        history["epoch"],
        history["val_accuracy"]
        * 100,
        label="Validation"
    )

    axes[0].set_xlabel(
        "Epoch"
    )

    axes[0].set_ylabel(
        "Accuracy (%)"
    )

    axes[0].set_title(
        "Training / Validation Accuracy"
    )

    axes[0].legend()

    axes[0].grid(
        alpha=0.3
    )

    axes[1].plot(
        history["epoch"],
        history["class_loss"],
        label="Class Loss"
    )

    axes[1].plot(
        history["epoch"],
        history["domain_loss"],
        label="Domain Loss"
    )

    axes[1].set_xlabel(
        "Epoch"
    )

    axes[1].set_ylabel(
        "Loss"
    )

    axes[1].set_title(
        "Training Loss"
    )

    axes[1].legend()

    axes[1].grid(
        alpha=0.3
    )

    plt.tight_layout()

    plt.savefig(
        RESULTS_DIR
        / "training_curves.png",
        dpi=200,
        bbox_inches="tight"
    )

    plt.close()


# ============================================================
# MAIN
# ============================================================

def main():

    print(
        "\n"
        + "=" * 70
    )

    print(
        "LOADING DATA"
    )

    print(
        "=" * 70
    )

    if not DATASET_DIR.exists():

        raise FileNotFoundError(
            f"Dataset directory does not exist:\n"
            f"{DATASET_DIR}"
        )

    # --------------------------------------------------------
    # Load all labeled T files
    # --------------------------------------------------------

    X, y, subject_ids = (
        load_all_subjects()
    )

    print(
        "\n"
        + "=" * 70
    )

    print(
        "COMPLETE DATASET"
    )

    print(
        "=" * 70
    )

    print(
        f"X: {X.shape}"
    )

    print(
        f"y: {y.shape}"
    )

    print(
        f"Classes: {np.bincount(y)}"
    )

    # --------------------------------------------------------
    # Create 50-trial FINAL holdout
    # --------------------------------------------------------

    (
        X_remaining,
        y_remaining,
        subject_remaining,
        X_test,
        y_test,
        subject_test
    ) = create_50_trial_test_set(
        X,
        y,
        subject_ids
    )

    # --------------------------------------------------------
    # Split remaining data into train/validation
    # --------------------------------------------------------

    (
        X_train,
        y_train,
        subject_train,
        X_val,
        y_val,
        subject_val
    ) = create_train_val_split(
        X_remaining,
        y_remaining,
        subject_remaining
    )

    print(
        "\n"
        + "=" * 70
    )

    print(
        "DATA SPLIT"
    )

    print(
        "=" * 70
    )

    print(
        f"Training   : {len(y_train)}"
    )

    print(
        f"Validation : {len(y_val)}"
    )

    print(
        f"Final test : {len(y_test)}"
    )

    print(
        f"Total      : "
        f"{len(y_train) + len(y_val) + len(y_test)}"
    )

    print(
        "\nTraining classes:"
    )

    print(
        np.bincount(y_train)
    )

    print(
        "Validation classes:"
    )

    print(
        np.bincount(y_val)
    )

    print(
        "Test classes:"
    )

    print(
        np.bincount(y_test)
    )

    # --------------------------------------------------------
    # Normalize using TRAINING statistics only
    # --------------------------------------------------------

    print(
        "\n"
        + "=" * 70
    )

    print(
        "NORMALIZATION"
    )

    print(
        "=" * 70
    )

    (
        X_train,
        X_val,
        X_test,
        train_mean,
        train_std
    ) = normalize_data(
        X_train,
        X_val,
        X_test
    )

    print(
        f"Training mean: "
        f"{train_mean:.8f}"
    )

    print(
        f"Training std : "
        f"{train_std:.8f}"
    )

    print(
        "\nValidation/test were normalized "
        "using training statistics only."
    )

    # --------------------------------------------------------
    # Train
    # --------------------------------------------------------

    (
        model,
        history,
        best_val_accuracy,
        best_epoch
    ) = train_model(
        X_train,
        y_train,
        subject_train,
        X_val,
        y_val,
        subject_val
    )

    plot_training_history(
        history
    )

    # --------------------------------------------------------
    # Final 50-trial test
    # --------------------------------------------------------

    test_metrics, test_predictions, test_probabilities = (
        evaluate_50_trial_test(
            model,
            X_test,
            y_test,
            subject_test
        )
    )

    # --------------------------------------------------------
    # Per-subject evaluation
    #
    # IMPORTANT:
    # This evaluates the 50-trial holdout by subject.
    # It does NOT train again.
    # --------------------------------------------------------

    print(
        "\n"
        + "=" * 70
    )

    print(
        "50-TRIAL TEST BY SUBJECT"
    )

    print(
        "=" * 70
    )

    test_subject_results = []

    for subject in SUBJECTS:

        mask = (
            subject_test
            == subject
        )

        if np.sum(mask) == 0:
            continue

        subject_y = (
            y_test[mask]
        )

        subject_pred = (
            test_predictions[mask]
        )

        metrics = calculate_metrics(
            subject_y,
            subject_pred
        )

        test_subject_results.append({

            "subject":
                subject,

            "test_trials":
                len(subject_y),

            "accuracy":
                metrics["accuracy"],

            "precision":
                metrics["precision"],

            "recall":
                metrics["recall"],

            "f1":
                metrics["f1"],

            "kappa":
                metrics["kappa"]
        })

        print(
            f"\nSubject {subject:02d}: "
            f"{len(subject_y)} test trials"
        )

        print(
            f"Accuracy: "
            f"{metrics['accuracy'] * 100:.2f}%"
        )

    test_subject_df = pd.DataFrame(
        test_subject_results
    )

    test_subject_df.to_csv(
        RESULTS_DIR
        / "50_trial_subject_results.csv",
        index=False
    )

    # --------------------------------------------------------
    # Final summary
    # --------------------------------------------------------

    print(
        "\n"
        + "=" * 70
    )

    print(
        "FINAL EXPERIMENT SUMMARY"
    )

    print(
        "=" * 70
    )

    print(
        f"Total labeled trials : "
        f"{len(y)}"
    )

    print(
        f"Training trials      : "
        f"{len(y_train)}"
    )

    print(
        f"Validation trials    : "
        f"{len(y_val)}"
    )

    print(
        f"Final test trials    : "
        f"{len(y_test)}"
    )

    print(
        f"Best validation acc  : "
        f"{best_val_accuracy * 100:.2f}%"
    )

    print(
        f"Best epoch           : "
        f"{best_epoch}"
    )

    print(
        f"FINAL 50-TRIAL ACC   : "
        f"{test_metrics['accuracy'] * 100:.2f}%"
    )

    print(
        f"FINAL 50-TRIAL F1   : "
        f"{test_metrics['f1'] * 100:.2f}%"
    )

    print(
        f"FINAL 50-TRIAL KAPPA: "
        f"{test_metrics['kappa']:.4f}"
    )

    print(
        "\n"
        + "=" * 70
    )

    print(
        "FILES CREATED"
    )

    print(
        "=" * 70
    )

    files = [

        "training_history.csv",

        "training_curves.png",

        "50_trial_predictions.csv",

        "50_trial_metrics.csv",

        "50_trial_confusion_matrix.png",

        "50_trial_subject_results.csv",

        "per_subject_results.csv",

        "per_subject_summary.csv",
    ]

    for filename in files:

        print(
            RESULTS_DIR
            / filename
        )

    print(
        "\nModel:"
    )

    print(
        MODEL_PATH
    )

    print(
        "\n"
        + "=" * 70
    )

    print(
        "EXPERIMENT COMPLETED"
    )

    print(
        "=" * 70
    )


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":

    main()
