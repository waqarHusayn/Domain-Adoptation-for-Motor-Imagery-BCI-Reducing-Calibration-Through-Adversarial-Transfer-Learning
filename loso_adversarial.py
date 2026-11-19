import os
import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    cohen_kappa_score,
    confusion_matrix
)

from dataset import load_subject


# ============================================================
# CONFIGURATION
# ============================================================

DATA_DIR = "BCICIV_2b_gdf"

SUBJECTS = list(range(1, 10))

EPOCHS = 30
BATCH_SIZE = 32

LEARNING_RATE = 0.001
WEIGHT_DECAY = 1e-4

DOMAIN_LOSS_WEIGHT = 0.1

SEED = 42

MODEL_DIR = "loso_models"
RESULTS_FILE = "loso_results.csv"


# ============================================================
# REPRODUCIBILITY
# ============================================================

def set_seed(seed=42):

    random.seed(seed)

    np.random.seed(seed)

    torch.manual_seed(seed)

    if torch.cuda.is_available():

        torch.cuda.manual_seed_all(seed)

        torch.backends.cudnn.deterministic = True

        torch.backends.cudnn.benchmark = False


set_seed(SEED)


# ============================================================
# DEVICE
# ============================================================

device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

print("=" * 80)
print("LEAVE-ONE-SUBJECT-OUT ADVERSARIAL TRANSFER LEARNING")
print("=" * 80)

print("Device:", device)

if torch.cuda.is_available():

    print(
        "GPU:",
        torch.cuda.get_device_name(0)
    )


# ============================================================
# GRADIENT REVERSAL LAYER
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


class GradientReversalLayer(nn.Module):

    def __init__(
        self,
        lambda_value=1.0
    ):

        super().__init__()

        self.lambda_value = lambda_value

    def forward(self, x):

        return GradientReversalFunction.apply(
            x,
            self.lambda_value
        )


# ============================================================
# EEGNET FEATURE EXTRACTOR
# ============================================================

class EEGNetFeatures(nn.Module):

    def __init__(
        self,
        channels=3,
        samples=1000
    ):

        super().__init__()

        self.features = nn.Sequential(

            # Temporal convolution
            nn.Conv2d(
                1,
                16,
                kernel_size=(1, 64),
                padding=(0, 32),
                bias=False
            ),

            nn.BatchNorm2d(16),

            # Spatial convolution
            nn.Conv2d(
                16,
                32,
                kernel_size=(channels, 1),
                groups=16,
                bias=False
            ),

            nn.BatchNorm2d(32),

            nn.ELU(),

            nn.AvgPool2d(
                kernel_size=(1, 4)
            ),

            nn.Dropout(0.25),

            # Separable temporal convolution
            nn.Conv2d(
                32,
                32,
                kernel_size=(1, 16),
                padding=(0, 8),
                groups=32,
                bias=False
            ),

            nn.Conv2d(
                32,
                32,
                kernel_size=(1, 1),
                bias=False
            ),

            nn.BatchNorm2d(32),

            nn.ELU(),

            nn.AvgPool2d(
                kernel_size=(1, 8)
            ),

            nn.Dropout(0.25)
        )

        # Determine flattened feature dimension
        with torch.no_grad():

            dummy = torch.zeros(
                1,
                1,
                channels,
                samples
            )

            output = self.features(dummy)

            self.feature_dim = (
                output.view(
                    1,
                    -1
                ).shape[1]
            )

    def forward(self, x):

        x = self.features(x)

        x = x.reshape(
            x.size(0),
            -1
        )

        return x


# ============================================================
# ADVERSARIAL EEGNET
# ============================================================

class AdversarialEEGNet(nn.Module):

    def __init__(
        self,
        channels=3,
        samples=1000,
        num_classes=2,
        lambda_value=1.0
    ):

        super().__init__()

        self.feature_extractor = EEGNetFeatures(
            channels=channels,
            samples=samples
        )

        feature_dim = (
            self.feature_extractor.feature_dim
        )

        print(
            "EEGNet feature dimension:",
            feature_dim
        )

        # Motor imagery classifier
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

        # Domain classifier
        self.domain_classifier = nn.Sequential(

            nn.Linear(
                feature_dim,
                64
            ),

            nn.ELU(),

            nn.Dropout(0.5),

            nn.Linear(
                64,
                2
            )
        )

        self.grl = GradientReversalLayer(
            lambda_value
        )

    def forward(
        self,
        x,
        return_domain=True
    ):

        features = (
            self.feature_extractor(x)
        )

        class_output = (
            self.classifier(features)
        )

        if return_domain:

            reversed_features = self.grl(
                features
            )

            domain_output = (
                self.domain_classifier(
                    reversed_features
                )
            )

            return (
                class_output,
                domain_output
            )

        return class_output


# ============================================================
# LOAD SUBJECT
# ============================================================

def load_subject_data(subject):

    print(
        f"\nLoading Subject {subject:02d}"
    )

    X, y = load_subject(
        subject,
        DATA_DIR
    )

    print(
        f"Subject {subject:02d}: "
        f"{X.shape}"
    )

    return X.astype(
        np.float32
    ), y.astype(
        np.int64
    )


# ============================================================
# NORMALIZATION
# ============================================================

def normalize_source_target(
    X_source,
    X_target
):

    # Normalize using SOURCE statistics only.
    #
    # This avoids using target-domain distribution
    # to calculate normalization statistics.

    mean = X_source.mean()

    std = X_source.std()

    if std < 1e-8:

        std = 1.0

    X_source = (
        X_source - mean
    ) / std

    X_target = (
        X_target - mean
    ) / std

    return (
        X_source.astype(np.float32),
        X_target.astype(np.float32)
    )


# ============================================================
# CREATE DATA LOADERS
# ============================================================

def create_loaders(
    X_source,
    y_source,
    X_target
):

    source_dataset = TensorDataset(

        torch.tensor(
            X_source,
            dtype=torch.float32
        ).unsqueeze(1),

        torch.tensor(
            y_source,
            dtype=torch.long
        )
    )

    target_dataset = TensorDataset(

        torch.tensor(
            X_target,
            dtype=torch.float32
        ).unsqueeze(1)
    )

    source_loader = DataLoader(

        source_dataset,

        batch_size=BATCH_SIZE,

        shuffle=True,

        drop_last=True
    )

    target_loader = DataLoader(

        target_dataset,

        batch_size=BATCH_SIZE,

        shuffle=True,

        drop_last=True
    )

    return (
        source_loader,
        target_loader
    )


# ============================================================
# TRAIN ONE LOSO EXPERIMENT
# ============================================================

def train_one_target(
    target_subject,
    subject_data
):

    source_subjects = [
        s
        for s in SUBJECTS
        if s != target_subject
    ]

    print("\n")
    print("#" * 80)
    print(
        f"TARGET SUBJECT: "
        f"{target_subject:02d}"
    )
    print(
        f"SOURCE SUBJECTS: "
        f"{source_subjects}"
    )
    print("#" * 80)

    # --------------------------------------------------------
    # Combine source subjects
    # --------------------------------------------------------

    X_source_list = []

    y_source_list = []

    for subject in source_subjects:

        Xs, ys = subject_data[
            subject
        ]

        X_source_list.append(Xs)

        y_source_list.append(ys)

    X_source = np.concatenate(
        X_source_list,
        axis=0
    )

    y_source = np.concatenate(
        y_source_list,
        axis=0
    )

    X_target, y_target = (
        subject_data[
            target_subject
        ]
    )

    print(
        "\nSource:",
        X_source.shape
    )

    print(
        "Target:",
        X_target.shape
    )

    # --------------------------------------------------------
    # Normalize
    # --------------------------------------------------------

    X_source, X_target = (
        normalize_source_target(
            X_source,
            X_target
        )
    )

    # --------------------------------------------------------
    # Loaders
    # --------------------------------------------------------

    source_loader, target_loader = (
        create_loaders(
            X_source,
            y_source,
            X_target
        )
    )

    # --------------------------------------------------------
    # Model
    # --------------------------------------------------------

    model = AdversarialEEGNet(
        channels=3,
        samples=1000,
        num_classes=2,
        lambda_value=1.0
    ).to(device)

    # --------------------------------------------------------
    # Loss functions
    # --------------------------------------------------------

    class_criterion = nn.CrossEntropyLoss()

    domain_criterion = nn.CrossEntropyLoss()

    # --------------------------------------------------------
    # Optimizer
    # --------------------------------------------------------

    optimizer = torch.optim.Adam(

        model.parameters(),

        lr=LEARNING_RATE,

        weight_decay=WEIGHT_DECAY
    )

    # --------------------------------------------------------
    # Training
    # --------------------------------------------------------

    best_loss = float("inf")

    best_state = None

    for epoch in range(
        1,
        EPOCHS + 1
    ):

        model.train()

        epoch_class_loss = 0.0

        epoch_domain_loss = 0.0

        epoch_total_loss = 0.0

        epoch_correct = 0

        epoch_total = 0

        target_iterator = iter(
            target_loader
        )

        for source_x, source_y in (
            source_loader
        ):

            try:

                target_x = next(
                    target_iterator
                )

            except StopIteration:

                target_iterator = iter(
                    target_loader
                )

                target_x = next(
                    target_iterator
                )

            target_x = target_x[0]

            source_x = source_x.to(
                device
            )

            source_y = source_y.to(
                device
            )

            target_x = target_x.to(
                device
            )

            # ------------------------------------------------
            # Source classification
            # ------------------------------------------------

            source_class_output, source_domain_output = (
                model(
                    source_x,
                    return_domain=True
                )
            )

            # ------------------------------------------------
            # Target domain prediction
            # ------------------------------------------------

            _, target_domain_output = (
                model(
                    target_x,
                    return_domain=True
                )
            )

            # Source = domain 0
            source_domain_labels = torch.zeros(
                source_x.size(0),
                dtype=torch.long,
                device=device
            )

            # Target = domain 1
            target_domain_labels = torch.ones(
                target_x.size(0),
                dtype=torch.long,
                device=device
            )

            # ------------------------------------------------
            # Losses
            # ------------------------------------------------

            classification_loss = (
                class_criterion(
                    source_class_output,
                    source_y
                )
            )

            source_domain_loss = (
                domain_criterion(
                    source_domain_output,
                    source_domain_labels
                )
            )

            target_domain_loss = (
                domain_criterion(
                    target_domain_output,
                    target_domain_labels
                )
            )

            domain_loss = (
                source_domain_loss
                +
                target_domain_loss
            ) / 2.0

            total_loss = (
                classification_loss
                +
                DOMAIN_LOSS_WEIGHT
                * domain_loss
            )

            # ------------------------------------------------
            # Backpropagation
            # ------------------------------------------------

            optimizer.zero_grad()

            total_loss.backward()

            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=5.0
            )

            optimizer.step()

            # ------------------------------------------------
            # Statistics
            # ------------------------------------------------

            epoch_class_loss += (
                classification_loss.item()
            )

            epoch_domain_loss += (
                domain_loss.item()
            )

            epoch_total_loss += (
                total_loss.item()
            )

            predictions = (
                source_class_output
                .argmax(dim=1)
            )

            epoch_correct += (
                predictions == source_y
            ).sum().item()

            epoch_total += (
                source_y.size(0)
            )

        # ----------------------------------------------------
        # Epoch statistics
        # ----------------------------------------------------

        source_accuracy = (
            epoch_correct /
            max(epoch_total, 1)
        )

        average_class_loss = (
            epoch_class_loss /
            len(source_loader)
        )

        average_domain_loss = (
            epoch_domain_loss /
            len(source_loader)
        )

        average_total_loss = (
            epoch_total_loss /
            len(source_loader)
        )

        print(
            f"Epoch {epoch:02d}/{EPOCHS} | "
            f"Total Loss: "
            f"{average_total_loss:.4f} | "
            f"Class Loss: "
            f"{average_class_loss:.4f} | "
            f"Domain Loss: "
            f"{average_domain_loss:.4f} | "
            f"Source Acc: "
            f"{source_accuracy * 100:.2f}%"
        )

        # ----------------------------------------------------
        # Save best model based on training loss
        # ----------------------------------------------------

        if average_total_loss < best_loss:

            best_loss = (
                average_total_loss
            )

            best_state = {
                k: v.detach().cpu().clone()
                for k, v in model.state_dict().items()
            }

    # --------------------------------------------------------
    # Restore best model
    # --------------------------------------------------------

    if best_state is not None:

        model.load_state_dict(
            best_state
        )

    # ========================================================
    # TARGET EVALUATION
    # ========================================================

    model.eval()

    target_tensor = torch.tensor(
        X_target,
        dtype=torch.float32
    ).unsqueeze(1)

    target_predictions = []

    with torch.no_grad():

        for start in range(
            0,
            len(target_tensor),
            BATCH_SIZE
        ):

            end = min(
                start + BATCH_SIZE,
                len(target_tensor)
            )

            batch = target_tensor[
                start:end
            ].to(device)

            class_output = model(
                batch,
                return_domain=False
            )

            predictions = (
                class_output
                .argmax(dim=1)
                .cpu()
                .numpy()
            )

            target_predictions.extend(
                predictions
            )

    target_predictions = np.asarray(
        target_predictions
    )

    # ========================================================
    # METRICS
    # ========================================================

    accuracy = accuracy_score(
        y_target,
        target_predictions
    )

    precision = precision_score(
        y_target,
        target_predictions,
        average="binary",
        zero_division=0
    )

    recall = recall_score(
        y_target,
        target_predictions,
        average="binary",
        zero_division=0
    )

    f1 = f1_score(
        y_target,
        target_predictions,
        average="binary",
        zero_division=0
    )

    kappa = cohen_kappa_score(
        y_target,
        target_predictions
    )

    cm = confusion_matrix(
        y_target,
        target_predictions
    )

    # ========================================================
    # PRINT RESULTS
    # ========================================================

    print("\n")
    print("-" * 80)

    print(
        f"TARGET SUBJECT {target_subject:02d} RESULTS"
    )

    print("-" * 80)

    print(
        f"Accuracy : "
        f"{accuracy * 100:.2f}%"
    )

    print(
        f"Precision: "
        f"{precision * 100:.2f}%"
    )

    print(
        f"Recall   : "
        f"{recall * 100:.2f}%"
    )

    print(
        f"F1       : "
        f"{f1 * 100:.2f}%"
    )

    print(
        f"Kappa    : "
        f"{kappa:.4f}"
    )

    print("\nConfusion matrix:")

    print(cm)

    # ========================================================
    # SAVE MODEL
    # ========================================================

    os.makedirs(
        MODEL_DIR,
        exist_ok=True
    )

    model_path = os.path.join(
        MODEL_DIR,
        f"target_subject_{target_subject:02d}.pth"
    )

    torch.save(
        {
            "model_state_dict":
                model.state_dict(),

            "target_subject":
                target_subject,

            "source_subjects":
                source_subjects,

            "accuracy":
                accuracy,

            "precision":
                precision,

            "recall":
                recall,

            "f1":
                f1,

            "kappa":
                kappa,

            "confusion_matrix":
                cm
        },
        model_path
    )

    print(
        "\nModel saved:",
        model_path
    )

    return {
        "target_subject":
            target_subject,

        "accuracy":
            accuracy,

        "precision":
            precision,

        "recall":
            recall,

        "f1":
            f1,

        "kappa":
            kappa
    }


# ============================================================
# MAIN
# ============================================================

def main():

    # --------------------------------------------------------
    # Load all subjects ONCE
    # --------------------------------------------------------

    print("\n")
    print("=" * 80)
    print("LOADING ALL SUBJECTS")
    print("=" * 80)

    subject_data = {}

    for subject in SUBJECTS:

        X, y = load_subject_data(
            subject
        )

        subject_data[
            subject
        ] = (X, y)

    # --------------------------------------------------------
    # Run LOSO
    # --------------------------------------------------------

    results = []

    for target_subject in SUBJECTS:

        result = train_one_target(
            target_subject,
            subject_data
        )

        results.append(
            result
        )

    # --------------------------------------------------------
    # Final results
    # --------------------------------------------------------

    print("\n")
    print("=" * 80)
    print("FINAL LOSO RESULTS")
    print("=" * 80)

    print(
        "\n"
        "Subject | Accuracy | Precision | Recall | F1 | Kappa"
    )

    print("-" * 80)

    for result in results:

        print(
            f"{result['target_subject']:7d} | "
            f"{result['accuracy'] * 100:8.2f}% | "
            f"{result['precision'] * 100:9.2f}% | "
            f"{result['recall'] * 100:6.2f}% | "
            f"{result['f1'] * 100:5.2f}% | "
            f"{result['kappa']:.4f}"
        )

    # --------------------------------------------------------
    # Calculate mean and standard deviation
    # --------------------------------------------------------

    accuracies = np.array([
        r["accuracy"]
        for r in results
    ])

    precisions = np.array([
        r["precision"]
        for r in results
    ])

    recalls = np.array([
        r["recall"]
        for r in results
    ])

    f1_scores = np.array([
        r["f1"]
        for r in results
    ])

    kappas = np.array([
        r["kappa"]
        for r in results
    ])

    print("\n")
    print("=" * 80)
    print("MEAN ± STANDARD DEVIATION")
    print("=" * 80)

    print(
        f"Accuracy : "
        f"{accuracies.mean() * 100:.2f}% "
        f"± "
        f"{accuracies.std() * 100:.2f}%"
    )

    print(
        f"Precision: "
        f"{precisions.mean() * 100:.2f}% "
        f"± "
        f"{precisions.std() * 100:.2f}%"
    )

    print(
        f"Recall   : "
        f"{recalls.mean() * 100:.2f}% "
        f"± "
        f"{recalls.std() * 100:.2f}%"
    )

    print(
        f"F1       : "
        f"{f1_scores.mean() * 100:.2f}% "
        f"± "
        f"{f1_scores.std() * 100:.2f}%"
    )

    print(
        f"Kappa    : "
        f"{kappas.mean():.4f} "
        f"± "
        f"{kappas.std():.4f}"
    )

    # --------------------------------------------------------
    # Save CSV
    # --------------------------------------------------------

    with open(
        RESULTS_FILE,
        "w"
    ) as f:

        f.write(
            "subject,accuracy,precision,"
            "recall,f1,kappa\n"
        )

        for result in results:

            f.write(
                f"{result['target_subject']},"
                f"{result['accuracy']:.6f},"
                f"{result['precision']:.6f},"
                f"{result['recall']:.6f},"
                f"{result['f1']:.6f},"
                f"{result['kappa']:.6f}\n"
            )

    print(
        "\nResults saved to:",
        RESULTS_FILE
    )

    print("\n")
    print("=" * 80)
    print("ALL LOSO EXPERIMENTS COMPLETE")
    print("=" * 80)


if __name__ == "__main__":

    main()
