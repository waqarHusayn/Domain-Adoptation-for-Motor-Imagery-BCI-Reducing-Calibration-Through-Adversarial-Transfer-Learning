from pathlib import Path

import mne
import numpy as np
import torch
from scipy.signal import butter, sosfiltfilt
from torch.utils.data import TensorDataset, DataLoader


# ============================================================
# CONFIGURATION
# ============================================================

DATA_DIR = Path("BCICIV_2b_gdf")

# Actual EEG channel names in the GDF files
EEG_CHANNELS = [
    "EEG:C3",
    "EEG:Cz",
    "EEG:C4"
]

SFREQ = 250.0

# Original BCI Competition IV 2b annotation codes
#
# 769 = left hand
# 770 = right hand
#
# MNE converts these to its own integer event IDs,
# so we obtain their actual IDs from event_dict.
LEFT_EVENT = "769"
RIGHT_EVENT = "770"


# ============================================================
# BAND-PASS FILTER
# ============================================================

def bandpass_filter(
    data,
    low=4.0,
    high=40.0,
    fs=250.0,
    order=4
):
    """
    Band-pass filter EEG.

    Input:
        data = channels x samples

    Output:
        filtered_data = channels x samples
    """

    sos = butter(
        order,
        [low, high],
        btype="bandpass",
        fs=fs,
        output="sos"
    )

    return sosfiltfilt(
        sos,
        data,
        axis=-1
    )


# ============================================================
# LOAD ONE GDF FILE
# ============================================================

def load_gdf_file(filepath):

    print("\n" + "=" * 60)
    print(f"Loading: {filepath}")
    print("=" * 60)

    # --------------------------------------------------------
    # Read GDF
    # --------------------------------------------------------

    raw = mne.io.read_raw_gdf(
        filepath,
        preload=True,
        verbose="ERROR"
    )

    print("\nChannels:")
    print(raw.ch_names)

    # --------------------------------------------------------
    # Verify EEG channels
    # --------------------------------------------------------

    missing = [
        ch for ch in EEG_CHANNELS
        if ch not in raw.ch_names
    ]

    if missing:

        raise RuntimeError(
            f"\nMissing EEG channels: {missing}\n"
            f"\nAvailable channels:\n{raw.ch_names}"
        )

    # --------------------------------------------------------
    # Select EEG channels
    # --------------------------------------------------------

    raw.pick(EEG_CHANNELS)

    print("\nSelected EEG channels:")
    print(raw.ch_names)

    # --------------------------------------------------------
    # Get EEG data
    # --------------------------------------------------------

    data = raw.get_data()

    print(
        "\nRaw EEG shape:",
        data.shape
    )

    # --------------------------------------------------------
    # Band-pass filter
    # --------------------------------------------------------

    print(
        "\nApplying 4-40 Hz band-pass filter..."
    )

    data = bandpass_filter(
        data,
        low=4.0,
        high=40.0,
        fs=SFREQ
    )

    # --------------------------------------------------------
    # Read annotations
    # --------------------------------------------------------

    print(
        "\nReading annotations..."
    )

    events, event_dict = mne.events_from_annotations(
        raw,
        verbose=False
    )

    print("\nEvent dictionary:")
    print(event_dict)

    print("\nFirst 30 events:")
    print(events[:30])

    print(
        "\nTotal events:",
        len(events)
    )

    # --------------------------------------------------------
    # Check that 769 and 770 exist
    # --------------------------------------------------------

    if LEFT_EVENT not in event_dict:

        raise RuntimeError(
            f"Event {LEFT_EVENT} was not found.\n"
            f"Available events: {list(event_dict.keys())}"
        )

    if RIGHT_EVENT not in event_dict:

        raise RuntimeError(
            f"Event {RIGHT_EVENT} was not found.\n"
            f"Available events: {list(event_dict.keys())}"
        )

    # --------------------------------------------------------
    # IMPORTANT
    #
    # event_dict contains:
    #
    # '769' -> 10
    # '770' -> 11
    #
    # We use these actual MNE IDs below.
    # --------------------------------------------------------

    left_id = event_dict[LEFT_EVENT]
    right_id = event_dict[RIGHT_EVENT]

    print("\nMotor imagery event IDs:")
    print(
        f"{LEFT_EVENT} -> MNE ID {left_id}"
    )
    print(
        f"{RIGHT_EVENT} -> MNE ID {right_id}"
    )

    # --------------------------------------------------------
    # Event counts
    # --------------------------------------------------------

    print("\nEvent counts:")

    for event_id, count in zip(
        *np.unique(
            events[:, 2],
            return_counts=True
        )
    ):
        print(
            f"MNE event {event_id}: "
            f"{count} occurrences"
        )

    # --------------------------------------------------------
    # Extract motor imagery trials
    # --------------------------------------------------------

    X = []
    y = []

    # --------------------------------------------------------
    # Trial timing
    #
    # BCI IV 2b:
    #
    # cue occurs at event
    #
    # We use:
    #
    # 0.5 sec -> 4.5 sec
    #
    # = 4 seconds of EEG
    #
    # 4 sec * 250 Hz = 1000 samples
    # --------------------------------------------------------

    start_offset = int(
        0.5 * SFREQ
    )

    end_offset = int(
        4.5 * SFREQ
    )

    print(
        "\nExtracting motor imagery trials..."
    )

    for event in events:

        sample = int(event[0])

        event_id = int(event[2])

        # ----------------------------------------------------
        # Left hand
        # ----------------------------------------------------

        if event_id == left_id:

            label = 0

        # ----------------------------------------------------
        # Right hand
        # ----------------------------------------------------

        elif event_id == right_id:

            label = 1

        # ----------------------------------------------------
        # Ignore other events
        # ----------------------------------------------------

        else:

            continue

        # ----------------------------------------------------
        # Trial window
        # ----------------------------------------------------

        start = (
            sample +
            start_offset
        )

        end = (
            sample +
            end_offset
        )

        # ----------------------------------------------------
        # Boundary check
        # ----------------------------------------------------

        if start < 0:
            continue

        if end > data.shape[1]:
            continue

        trial = data[
            :,
            start:end
        ]

        # ----------------------------------------------------
        # Verify trial shape
        # ----------------------------------------------------

        if trial.shape != (
            3,
            1000
        ):
            continue

        X.append(trial)

        y.append(label)

    # --------------------------------------------------------
    # Verify extraction
    # --------------------------------------------------------

    if len(X) == 0:

        raise RuntimeError(
            "No motor imagery trials were extracted."
        )

    # --------------------------------------------------------
    # Convert to numpy
    # --------------------------------------------------------

    X = np.asarray(
        X,
        dtype=np.float32
    )

    y = np.asarray(
        y,
        dtype=np.int64
    )

    print(
        "\nSuccessfully extracted trials!"
    )

    print(
        "X shape:",
        X.shape
    )

    print(
        "y shape:",
        y.shape
    )

    print(
        "Class distribution:",
        np.bincount(y)
    )

    return X, y


# ============================================================
# NORMALIZATION
# ============================================================

def normalize_trials(X):

    """
    Z-score each EEG channel independently
    for each trial.
    """

    mean = X.mean(
        axis=-1,
        keepdims=True
    )

    std = X.std(
        axis=-1,
        keepdims=True
    )

    std = np.maximum(
        std,
        1e-6
    )

    X = (
        X - mean
    ) / std

    return X.astype(
        np.float32
    )


# ============================================================
# LOAD SUBJECT
# ============================================================

def load_subject(subject):

    """
    Load training sessions for one subject.

    Example:

        subject=1

    loads:

        B0101T.gdf
        B0102T.gdf
        B0103T.gdf
    """

    subject = int(subject)

    X_all = []
    y_all = []

    print("\n")
    print("#" * 60)
    print(
        f"LOADING SUBJECT {subject:02d}"
    )
    print("#" * 60)

    # --------------------------------------------------------
    # Training sessions
    # --------------------------------------------------------

    for session in [1, 2, 3]:

        filename = (
            f"B{subject:02d}"
            f"{session:02d}T.gdf"
        )

        filepath = (
            DATA_DIR /
            filename
        )

        if not filepath.exists():

            print(
                f"\nWARNING: "
                f"{filepath} not found"
            )

            continue

        X, y = load_gdf_file(
            filepath
        )

        X_all.append(X)
        y_all.append(y)

    # --------------------------------------------------------
    # Verify
    # --------------------------------------------------------

    if not X_all:

        raise RuntimeError(
            f"No training files found "
            f"for subject {subject}."
        )

    # --------------------------------------------------------
    # Combine sessions
    # --------------------------------------------------------

    X = np.concatenate(
        X_all,
        axis=0
    )

    y = np.concatenate(
        y_all,
        axis=0
    )

    print("\nCombined subject data:")

    print(
        "X:",
        X.shape
    )

    print(
        "y:",
        y.shape
    )

    # --------------------------------------------------------
    # Normalize
    # --------------------------------------------------------

    X = normalize_trials(X)

    print("\nNormalized data:")

    print(
        "Mean:",
        X.mean()
    )

    print(
        "Std:",
        X.std()
    )

    print(
        "\nTotal trials:",
        len(X)
    )

    print(
        "Class distribution:",
        np.bincount(y)
    )

    return X, y


# ============================================================
# PYTORCH DATALOADER
# ============================================================

def make_loader(
    X,
    y,
    batch_size=32,
    shuffle=True
):

    """
    Convert:

        N x 3 x 1000

    into:

        N x 1 x 3 x 1000
    """

    X = X[:, np.newaxis, :, :]

    X = torch.tensor(
        X,
        dtype=torch.float32
    )

    y = torch.tensor(
        y,
        dtype=torch.long
    )

    dataset = TensorDataset(
        X,
        y
    )

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=False
    )

    return loader


# ============================================================
# MAIN TEST
# ============================================================

if __name__ == "__main__":

    print("=" * 60)
    print("BCI COMPETITION IV 2B DATASET TEST")
    print("=" * 60)

    print(
        "\nDataset directory:"
    )

    print(
        DATA_DIR.resolve()
    )

    # --------------------------------------------------------
    # Check directory
    # --------------------------------------------------------

    if not DATA_DIR.exists():

        raise FileNotFoundError(
            f"\nDataset directory not found:\n"
            f"{DATA_DIR.resolve()}"
        )

    # --------------------------------------------------------
    # Test Subject 1
    # --------------------------------------------------------

    X, y = load_subject(1)

    # --------------------------------------------------------
    # Final dataset
    # --------------------------------------------------------

    print("\n")
    print("=" * 60)
    print("FINAL DATASET")
    print("=" * 60)

    print(
        "X shape:",
        X.shape
    )

    print(
        "y shape:",
        y.shape
    )

    # --------------------------------------------------------
    # DataLoader
    # --------------------------------------------------------

    loader = make_loader(
        X,
        y,
        batch_size=32,
        shuffle=True
    )

    # --------------------------------------------------------
    # Test batch
    # --------------------------------------------------------

    X_batch, y_batch = next(
        iter(loader)
    )

    print("\n")
    print("=" * 60)
    print("PYTORCH BATCH")
    print("=" * 60)

    print(
        "X batch shape:",
        X_batch.shape
    )

    print(
        "y batch shape:",
        y_batch.shape
    )

    print(
        "X dtype:",
        X_batch.dtype
    )

    print(
        "y dtype:",
        y_batch.dtype
    )

    print("\n")
    print("=" * 60)
    print("DATASET TEST COMPLETED SUCCESSFULLY")
    print("=" * 60)
