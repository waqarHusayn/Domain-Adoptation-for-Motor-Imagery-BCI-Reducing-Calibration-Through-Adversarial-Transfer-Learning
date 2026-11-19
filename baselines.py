# ================================================================
# BCI COMPETITION IV 2B
# CLASSICAL BASELINE MODELS
#
# LOSO = Leave-One-Subject-Out
#
# Models:
#   1. LDA + Band Power
#   2. Logistic Regression + Band Power
#   3. Linear SVM + Band Power
#   4. RBF SVM + Band Power
#   5. Random Forest + Band Power
#   6. CSP + LDA
#
# Dataset:
#   BCI Competition IV Dataset 2b
#
# EEG:
#   C3, Cz, C4
#
# Preprocessing:
#   4-40 Hz band-pass
#   4 second trial
#
# Evaluation:
#   Leave-One-Subject-Out
#
# ================================================================

import os
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import mne

from scipy.signal import welch

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    cohen_kappa_score,
    confusion_matrix,
)

from mne.decoding import CSP


warnings.filterwarnings("ignore")

# ================================================================
# CONFIGURATION
# ================================================================

DATASET_DIR = r"C:\Users\Student.DESKTOP-RUSD6ON.001\Desktop\FYP\BCICIV_2b_gdf"

RESULTS_DIR = os.path.join(
    os.path.dirname(DATASET_DIR),
    "results_baselines"
)

os.makedirs(RESULTS_DIR, exist_ok=True)

SUBJECTS = list(range(1, 10))

EEG_CHANNELS = [
    "EEG:C3",
    "EEG:Cz",
    "EEG:C4",
]

LOW_FREQ = 4.0
HIGH_FREQ = 40.0

SFREQ = 250.0

# Event codes used by BCI IV 2b
MI_LEFT = "769"
MI_RIGHT = "770"

RANDOM_STATE = 42


# ================================================================
# PRINT HEADER
# ================================================================

print("=" * 70)
print("BCI IV 2B CLASSICAL BASELINE EVALUATION")
print("=" * 70)

print(f"Dataset : {DATASET_DIR}")
print(f"Subjects: {SUBJECTS}")
print(f"Channels: {EEG_CHANNELS}")
print(f"Filter  : {LOW_FREQ}-{HIGH_FREQ} Hz")
print()


# ================================================================
# FIND GDF FILES
# ================================================================

def find_subject_files(subject):
    """
    Find all GDF files belonging to a subject.

    We primarily use training files (*T.gdf), because these contain
    the labeled motor imagery trials.

    Some files such as B0104E.gdf do not contain 769/770 labels,
    therefore they are intentionally not used.
    """

    subject_string = f"B{subject:02d}"

    files = []

    for filename in sorted(os.listdir(DATASET_DIR)):

        filename_upper = filename.upper()

        if (
            filename_upper.startswith(subject_string)
            and filename_upper.endswith("T.GDF")
        ):
            files.append(
                os.path.join(DATASET_DIR, filename)
            )

    return files


# ================================================================
# LOAD ONE GDF FILE
# ================================================================

def load_gdf_file(filepath):
    """
    Load one GDF file and extract motor imagery trials.

    Returns
    -------
    X : ndarray
        Shape = (trials, channels, samples)

    y : ndarray
        Binary labels:
        0 = event 769
        1 = event 770
    """

    print("-" * 70)
    print(f"Loading: {filepath}")
    print("-" * 70)

    # ------------------------------------------------------------
    # Read GDF
    # ------------------------------------------------------------

    raw = mne.io.read_raw_gdf(
        filepath,
        preload=True,
        verbose=False
    )

    print("Channels:")
    print(raw.ch_names)

    # ------------------------------------------------------------
    # Pick EEG channels
    # ------------------------------------------------------------

    available_channels = [
        ch for ch in EEG_CHANNELS
        if ch in raw.ch_names
    ]

    if len(available_channels) != 3:

        raise RuntimeError(
            f"Required EEG channels not found in {filepath}\n"
            f"Found: {raw.ch_names}"
        )

    raw.pick(available_channels)

    print()
    print("Selected EEG channels:")
    print(raw.ch_names)

    print()
    print("Raw EEG shape:", raw.get_data().shape)

    # ------------------------------------------------------------
    # Band-pass filter
    # ------------------------------------------------------------

    print()
    print(
        f"Applying {LOW_FREQ}-{HIGH_FREQ} Hz band-pass filter..."
    )

    raw.filter(
        l_freq=LOW_FREQ,
        h_freq=HIGH_FREQ,
        verbose=False
    )

    # ------------------------------------------------------------
    # Get events
    # ------------------------------------------------------------

    print()
    print("Reading annotations...")

    events, event_dict = mne.events_from_annotations(
        raw,
        verbose=False
    )

    print()
    print("Event dictionary:")
    print(event_dict)

    # ------------------------------------------------------------
    # Locate MI events
    # ------------------------------------------------------------

    if MI_LEFT not in event_dict or MI_RIGHT not in event_dict:

        raise RuntimeError(
            f"Could not find motor imagery events "
            f"{MI_LEFT}/{MI_RIGHT} in {filepath}"
        )

    left_id = event_dict[MI_LEFT]
    right_id = event_dict[MI_RIGHT]

    print()
    print(
        f"Motor imagery IDs: "
        f"{MI_LEFT} -> {left_id}, "
        f"{MI_RIGHT} -> {right_id}"
    )

    # ------------------------------------------------------------
    # Create event mapping
    # ------------------------------------------------------------

    event_mapping = {
        MI_LEFT: left_id,
        MI_RIGHT: right_id,
    }

    # ------------------------------------------------------------
    # Epoching
    #
    # Dataset event 769/770 occurs at the beginning of the
    # motor imagery period.
    #
    # We use 0-4 seconds.
    # ------------------------------------------------------------

    try:

        epochs = mne.Epochs(
            raw,
            events,
            event_id=event_mapping,
            tmin=0.0,
            tmax=3.996,
            baseline=None,
            preload=True,
            reject_by_annotation=True,
            verbose=False
        )

    except Exception as e:

        raise RuntimeError(
            f"Epoching failed for {filepath}: {e}"
        )

    # ------------------------------------------------------------
    # Extract arrays
    # ------------------------------------------------------------

    X = epochs.get_data(copy=True)

    event_labels = epochs.events[:, -1]

    y = np.zeros(
        len(event_labels),
        dtype=np.int64
    )

    y[event_labels == right_id] = 1

    # ------------------------------------------------------------
    # Verify
    # ------------------------------------------------------------

    print()
    print("Trials:", X.shape)
    print("Labels:", y.shape)

    print(
        "Class distribution:",
        np.bincount(y)
    )

    return X.astype(np.float32), y


# ================================================================
# LOAD ALL SUBJECTS
# ================================================================

def load_all_subjects():

    subject_data = {}

    print()
    print("=" * 70)
    print("STEP 1: LOADING DATA")
    print("=" * 70)

    for subject in SUBJECTS:

        print()
        print("#" * 70)
        print(f"LOADING SUBJECT {subject:02d}")
        print("#" * 70)

        files = find_subject_files(subject)

        if not files:

            print(
                f"WARNING: No training files found "
                f"for subject {subject:02d}"
            )

            continue

        subject_X = []
        subject_y = []

        for filepath in files:

            try:

                X, y = load_gdf_file(filepath)

                subject_X.append(X)
                subject_y.append(y)

            except RuntimeError as e:

                print()
                print("WARNING:")
                print(e)

                continue

        if not subject_X:

            print(
                f"WARNING: No labeled trials for "
                f"subject {subject:02d}"
            )

            continue

        X_subject = np.concatenate(
            subject_X,
            axis=0
        )

        y_subject = np.concatenate(
            subject_y,
            axis=0
        )

        subject_data[subject] = (
            X_subject,
            y_subject
        )

        print()
        print(
            f"Subject {subject:02d} summary:"
        )

        print(
            "X:",
            X_subject.shape
        )

        print(
            "y:",
            y_subject.shape
        )

        print(
            "Class distribution:",
            np.bincount(y_subject)
        )

    # ------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------

    total_trials = sum(
        len(y)
        for X, y in subject_data.values()
    )

    print()
    print("=" * 70)
    print("DATA LOADING COMPLETE")
    print("=" * 70)

    print(
        "Total labeled trials:",
        total_trials
    )

    return subject_data


# ================================================================
# BAND POWER FEATURES
# ================================================================

def extract_bandpower_features(
    X,
    sfreq=250.0
):
    """
    Extract log band-power features.

    Input:
        X = (trials, channels, samples)

    Output:
        features = (trials, features)

    Frequency bands:

        8-12   Hz  mu
        12-16  Hz
        16-20  Hz
        20-24  Hz
        24-30  Hz
        30-40  Hz

    3 channels × 6 bands = 18 features.
    """

    bands = [
        (8, 12),
        (12, 16),
        (16, 20),
        (20, 24),
        (24, 30),
        (30, 40),
    ]

    n_trials = X.shape[0]

    features = []

    for trial_idx in range(n_trials):

        trial_features = []

        for channel_idx in range(
            X.shape[1]
        ):

            signal = X[
                trial_idx,
                channel_idx
            ]

            freqs, psd = welch(
                signal,
                fs=sfreq,
                nperseg=min(
                    250,
                    len(signal)
                )
            )

            for low, high in bands:

                mask = (
                    (freqs >= low)
                    &
                    (freqs < high)
                )

                if np.any(mask):

                    power = np.trapezoid(
                        psd[mask],
                        freqs[mask]
                    )

                else:

                    power = 1e-12

                trial_features.append(
                    np.log(
                        power + 1e-12
                    )
                )

        features.append(
            trial_features
        )

    return np.asarray(
        features,
        dtype=np.float32
    )


# ================================================================
# MODEL DEFINITIONS
# ================================================================

def create_models():

    models = {}

    # ------------------------------------------------------------
    # LDA
    # ------------------------------------------------------------

    models["LDA"] = Pipeline([
        (
            "scaler",
            StandardScaler()
        ),
        (
            "classifier",
            LinearDiscriminantAnalysis()
        )
    ])

    # ------------------------------------------------------------
    # Logistic Regression
    # ------------------------------------------------------------

    models["Logistic Regression"] = Pipeline([
        (
            "scaler",
            StandardScaler()
        ),
        (
            "classifier",
            LogisticRegression(
                max_iter=2000,
                random_state=RANDOM_STATE
            )
        )
    ])

    # ------------------------------------------------------------
    # Linear SVM
    # ------------------------------------------------------------

    models["Linear SVM"] = Pipeline([
        (
            "scaler",
            StandardScaler()
        ),
        (
            "classifier",
            SVC(
                kernel="linear",
                C=1.0
            )
        )
    ])

    # ------------------------------------------------------------
    # RBF SVM
    # ------------------------------------------------------------

    models["RBF SVM"] = Pipeline([
        (
            "scaler",
            StandardScaler()
        ),
        (
            "classifier",
            SVC(
                kernel="rbf",
                C=1.0,
                gamma="scale"
            )
        )
    ])

    # ------------------------------------------------------------
    # Random Forest
    # ------------------------------------------------------------

    models["Random Forest"] = Pipeline([
        (
            "classifier",
            RandomForestClassifier(
                n_estimators=300,
                max_depth=None,
                min_samples_split=2,
                min_samples_leaf=1,
                random_state=RANDOM_STATE,
                n_jobs=-1
            )
        )
    ])

    return models


# ================================================================
# METRICS
# ================================================================

def calculate_metrics(
    y_true,
    y_pred
):

    return {
        "accuracy": accuracy_score(
            y_true,
            y_pred
        ),

        "precision": precision_score(
            y_true,
            y_pred,
            zero_division=0
        ),

        "recall": recall_score(
            y_true,
            y_pred,
            zero_division=0
        ),

        "f1": f1_score(
            y_true,
            y_pred,
            zero_division=0
        ),

        "kappa": cohen_kappa_score(
            y_true,
            y_pred
        )
    }


# ================================================================
# BASELINE LOSO
# ================================================================

def run_bandpower_loso(
    subject_data,
    model_name,
    model
):

    print()
    print("=" * 70)
    print(f"MODEL: {model_name}")
    print("=" * 70)

    results = []

    for test_subject in SUBJECTS:

        if test_subject not in subject_data:

            continue

        print()
        print(
            f"Testing subject "
            f"{test_subject:02d} "
            f"with {model_name}"
        )

        # --------------------------------------------------------
        # Training subjects
        # --------------------------------------------------------

        train_X_list = []
        train_y_list = []

        for subject in SUBJECTS:

            if subject == test_subject:
                continue

            if subject not in subject_data:
                continue

            X_subject, y_subject = (
                subject_data[subject]
            )

            train_X_list.append(
                X_subject
            )

            train_y_list.append(
                y_subject
            )

        X_train = np.concatenate(
            train_X_list,
            axis=0
        )

        y_train = np.concatenate(
            train_y_list,
            axis=0
        )

        # --------------------------------------------------------
        # Test subject
        # --------------------------------------------------------

        X_test, y_test = subject_data[
            test_subject
        ]

        print(
            "Training trials:",
            len(y_train)
        )

        print(
            "Testing trials :",
            len(y_test)
        )

        # --------------------------------------------------------
        # Feature extraction
        #
        # IMPORTANT:
        # Features are extracted independently.
        #
        # No scaler or statistical transformation is fitted
        # using the test subject.
        # --------------------------------------------------------

        print(
            "Extracting training features..."
        )

        X_train_features = (
            extract_bandpower_features(
                X_train,
                sfreq=SFREQ
            )
        )

        print(
            "Extracting testing features..."
        )

        X_test_features = (
            extract_bandpower_features(
                X_test,
                sfreq=SFREQ
            )
        )

        print(
            "Training feature shape:",
            X_train_features.shape
        )

        print(
            "Testing feature shape :",
            X_test_features.shape
        )

        # --------------------------------------------------------
        # Train
        # --------------------------------------------------------

        print("Training...")

        model.fit(
            X_train_features,
            y_train
        )

        # --------------------------------------------------------
        # Predict
        # --------------------------------------------------------

        y_pred = model.predict(
            X_test_features
        )

        # --------------------------------------------------------
        # Metrics
        # --------------------------------------------------------

        metrics = calculate_metrics(
            y_test,
            y_pred
        )

        cm = confusion_matrix(
            y_test,
            y_pred
        )

        print()
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

        print()
        print("Confusion matrix:")
        print(cm)

        # --------------------------------------------------------
        # Save result
        # --------------------------------------------------------

        results.append({
            "model": model_name,
            "subject": test_subject,
            "accuracy": metrics["accuracy"],
            "precision": metrics["precision"],
            "recall": metrics["recall"],
            "f1": metrics["f1"],
            "kappa": metrics["kappa"],
            "test_trials": len(y_test)
        })

    return results


# ================================================================
# CSP + LDA LOSO
# ================================================================

def run_csp_loso(subject_data):

    model_name = "CSP + LDA"

    print()
    print("=" * 70)
    print("MODEL: CSP + LDA")
    print("=" * 70)

    results = []

    for test_subject in SUBJECTS:

        if test_subject not in subject_data:
            continue

        print()
        print(
            f"Testing subject "
            f"{test_subject:02d} "
            f"with CSP + LDA"
        )

        # --------------------------------------------------------
        # Training data
        # --------------------------------------------------------

        train_X_list = []
        train_y_list = []

        for subject in SUBJECTS:

            if subject == test_subject:
                continue

            if subject not in subject_data:
                continue

            X_subject, y_subject = (
                subject_data[subject]
            )

            train_X_list.append(
                X_subject
            )

            train_y_list.append(
                y_subject
            )

        X_train = np.concatenate(
            train_X_list,
            axis=0
        )

        y_train = np.concatenate(
            train_y_list,
            axis=0
        )

        # --------------------------------------------------------
        # Test data
        # --------------------------------------------------------

        X_test, y_test = subject_data[
            test_subject
        ]

        print(
            "Training trials:",
            len(y_train)
        )

        print(
            "Testing trials :",
            len(y_test)
        )

        # --------------------------------------------------------
        # CSP
        #
        # CSP is fitted ONLY on training subjects.
        # --------------------------------------------------------

        csp = CSP(
            n_components=4,
            reg="ledoit_wolf",
            log=True,
            norm_trace=False
        )

        print("Fitting CSP...")

        X_train_csp = csp.fit_transform(
            X_train,
            y_train
        )

        print("Transforming test data...")

        X_test_csp = csp.transform(
            X_test
        )

        print(
            "CSP training shape:",
            X_train_csp.shape
        )

        print(
            "CSP testing shape :",
            X_test_csp.shape
        )

        # --------------------------------------------------------
        # LDA
        # --------------------------------------------------------

        lda = Pipeline([
            (
                "scaler",
                StandardScaler()
            ),
            (
                "classifier",
                LinearDiscriminantAnalysis()
            )
        ])

        print("Training LDA...")

        lda.fit(
            X_train_csp,
            y_train
        )

        y_pred = lda.predict(
            X_test_csp
        )

        # --------------------------------------------------------
        # Metrics
        # --------------------------------------------------------

        metrics = calculate_metrics(
            y_test,
            y_pred
        )

        cm = confusion_matrix(
            y_test,
            y_pred
        )

        print()
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

        print()
        print("Confusion matrix:")
        print(cm)

        results.append({
            "model": model_name,
            "subject": test_subject,
            "accuracy": metrics["accuracy"],
            "precision": metrics["precision"],
            "recall": metrics["recall"],
            "f1": metrics["f1"],
            "kappa": metrics["kappa"],
            "test_trials": len(y_test)
        })

    return results


# ================================================================
# SUMMARY
# ================================================================

def create_summary(results_df):

    print()
    print("=" * 70)
    print("BASELINE SUMMARY")
    print("=" * 70)

    summary_rows = []

    for model_name in results_df[
        "model"
    ].unique():

        data = results_df[
            results_df["model"]
            == model_name
        ]

        row = {
            "model": model_name,

            "accuracy_mean":
                data["accuracy"].mean(),

            "accuracy_std":
                data["accuracy"].std(
                    ddof=1
                ),

            "precision_mean":
                data["precision"].mean(),

            "precision_std":
                data["precision"].std(
                    ddof=1
                ),

            "recall_mean":
                data["recall"].mean(),

            "recall_std":
                data["recall"].std(
                    ddof=1
                ),

            "f1_mean":
                data["f1"].mean(),

            "f1_std":
                data["f1"].std(
                    ddof=1
                ),

            "kappa_mean":
                data["kappa"].mean(),

            "kappa_std":
                data["kappa"].std(
                    ddof=1
                )
        }

        summary_rows.append(row)

    summary_df = pd.DataFrame(
        summary_rows
    )

    print()

    for _, row in summary_df.iterrows():

        print(
            f"{row['model']}"
        )

        print(
            f"  Accuracy : "
            f"{row['accuracy_mean'] * 100:.2f} "
            f"+/- "
            f"{row['accuracy_std'] * 100:.2f}%"
        )

        print(
            f"  Precision: "
            f"{row['precision_mean'] * 100:.2f} "
            f"+/- "
            f"{row['precision_std'] * 100:.2f}%"
        )

        print(
            f"  Recall   : "
            f"{row['recall_mean'] * 100:.2f} "
            f"+/- "
            f"{row['recall_std'] * 100:.2f}%"
        )

        print(
            f"  F1       : "
            f"{row['f1_mean'] * 100:.2f} "
            f"+/- "
            f"{row['f1_std'] * 100:.2f}%"
        )

        print(
            f"  Kappa    : "
            f"{row['kappa_mean']:.4f} "
            f"+/- "
            f"{row['kappa_std']:.4f}"
        )

        print()

    return summary_df


# ================================================================
# PLOT MODEL COMPARISON
# ================================================================

def plot_model_comparison(
    summary_df
):

    models = summary_df[
        "model"
    ].values

    accuracy = (
        summary_df[
            "accuracy_mean"
        ].values
        * 100
    )

    accuracy_std = (
        summary_df[
            "accuracy_std"
        ].values
        * 100
    )

    plt.figure(
        figsize=(12, 7)
    )

    x = np.arange(
        len(models)
    )

    bars = plt.bar(
        x,
        accuracy,
        yerr=accuracy_std,
        capsize=5,
        color=[
            "#4C78A8",
            "#F58518",
            "#54A24B",
            "#E45756",
            "#72B7B2",
            "#B279A2"
        ],
        alpha=0.9
    )

    plt.xticks(
        x,
        models,
        rotation=20,
        ha="right"
    )

    plt.ylabel(
        "LOSO Accuracy (%)"
    )

    plt.xlabel(
        "Model"
    )

    plt.title(
        "BCI IV 2b Classical Baseline Comparison"
    )

    plt.ylim(
        0,
        100
    )

    plt.grid(
        axis="y",
        alpha=0.25
    )

    for bar, value in zip(
        bars,
        accuracy
    ):

        plt.text(
            bar.get_x()
            + bar.get_width() / 2,
            bar.get_height()
            + 1,
            f"{value:.2f}%",
            ha="center",
            va="bottom",
            fontsize=9
        )

    plt.tight_layout()

    output_path = os.path.join(
        RESULTS_DIR,
        "baseline_model_comparison.png"
    )

    plt.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    print()
    print(
        "Saved plot:",
        output_path
    )


# ================================================================
# PLOT SUBJECT RESULTS
# ================================================================

def plot_subject_accuracy(
    results_df
):

    models = results_df[
        "model"
    ].unique()

    subjects = sorted(
        results_df[
            "subject"
        ].unique()
    )

    plt.figure(
        figsize=(14, 8)
    )

    for model in models:

        data = results_df[
            results_df["model"]
            == model
        ]

        data = data.sort_values(
            "subject"
        )

        plt.plot(
            data["subject"],
            data["accuracy"] * 100,
            marker="o",
            linewidth=2,
            label=model
        )

    plt.xlabel(
        "Test Subject"
    )

    plt.ylabel(
        "Accuracy (%)"
    )

    plt.title(
        "LOSO Accuracy by Subject"
    )

    plt.xticks(
        subjects
    )

    plt.ylim(
        0,
        100
    )

    plt.grid(
        alpha=0.25
    )

    plt.legend()

    plt.tight_layout()

    output_path = os.path.join(
        RESULTS_DIR,
        "baseline_subject_accuracy.png"
    )

    plt.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    print(
        "Saved plot:",
        output_path
    )


# ================================================================
# SAVE CONFUSION MATRICES
# ================================================================

def calculate_overall_confusion_matrices(
    subject_data
):

    """
    This function is intentionally not used for model fitting.

    Individual subject confusion matrices are already printed.
    """

    pass


# ================================================================
# MAIN
# ================================================================

if __name__ == "__main__":

    # ------------------------------------------------------------
    # STEP 1
    # ------------------------------------------------------------

    subject_data = load_all_subjects()

    if len(subject_data) < 2:

        raise RuntimeError(
            "At least two subjects are required "
            "for LOSO evaluation."
        )

    # ------------------------------------------------------------
    # STEP 2
    # ------------------------------------------------------------

    print()
    print("=" * 70)
    print("STEP 2: RUNNING BASELINES")
    print("=" * 70)

    all_results = []

    # ------------------------------------------------------------
    # Band-power models
    # ------------------------------------------------------------

    models = create_models()

    for model_name, model in models.items():

        model_results = run_bandpower_loso(
            subject_data,
            model_name,
            model
        )

        all_results.extend(
            model_results
        )

    # ------------------------------------------------------------
    # CSP + LDA
    # ------------------------------------------------------------

    csp_results = run_csp_loso(
        subject_data
    )

    all_results.extend(
        csp_results
    )

    # ------------------------------------------------------------
    # Convert to DataFrame
    # ------------------------------------------------------------

    results_df = pd.DataFrame(
        all_results
    )

    # ------------------------------------------------------------
    # Save detailed results
    # ------------------------------------------------------------

    detailed_path = os.path.join(
        RESULTS_DIR,
        "baseline_loso_results.csv"
    )

    results_df.to_csv(
        detailed_path,
        index=False
    )

    print()
    print(
        "Detailed results saved:",
        detailed_path
    )

    # ------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------

    summary_df = create_summary(
        results_df
    )

    summary_path = os.path.join(
        RESULTS_DIR,
        "baseline_summary.csv"
    )

    summary_df.to_csv(
        summary_path,
        index=False
    )

    print(
        "Summary saved:",
        summary_path
    )

    # ------------------------------------------------------------
    # Plots
    # ------------------------------------------------------------

    plot_model_comparison(
        summary_df
    )

    plot_subject_accuracy(
        results_df
    )

    # ------------------------------------------------------------
    # Final output
    # ------------------------------------------------------------

    print()
    print("=" * 70)
    print("BASELINE EVALUATION COMPLETE")
    print("=" * 70)

    print()
    print(
        "Results directory:"
    )

    print(
        RESULTS_DIR
    )

    print()
    print(
        "Files created:"
    )

    print(
        " - baseline_loso_results.csv"
    )

    print(
        " - baseline_summary.csv"
    )

    print(
        " - baseline_model_comparison.png"
    )

    print(
        " - baseline_subject_accuracy.png"
    )

    print()
    print("=" * 70)
