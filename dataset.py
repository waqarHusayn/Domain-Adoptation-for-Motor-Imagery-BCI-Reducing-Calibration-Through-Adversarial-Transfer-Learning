import os
import numpy as np
import mne


# ============================================================
# CONFIGURATION
# ============================================================

DATA_DIR = "BCICIV_2b_gdf"

EEG_CHANNELS = [
    "EEG:C3",
    "EEG:Cz",
    "EEG:C4"
]

FS = 250

# 4 seconds after motor-imagery cue
TRIAL_SAMPLES = 1000

LOW_FREQ = 4
HIGH_FREQ = 40


# ============================================================
# FIND EEG CHANNELS
# ============================================================

def find_eeg_channels(raw):

    mapping = {}

    for ch in raw.ch_names:

        clean = ch.upper().replace("EEG:", "")

        if clean in ["C3", "CZ", "C4"]:

            mapping[clean] = ch

    required = ["C3", "CZ", "C4"]

    missing = [
        ch for ch in required
        if ch not in mapping
    ]

    if missing:

        raise RuntimeError(
            f"Missing EEG channels: {missing}\n"
            f"Available channels: {raw.ch_names}"
        )

    return [
        mapping["C3"],
        mapping["CZ"],
        mapping["C4"]
    ]


# ============================================================
# LOAD ONE GDF FILE
# ============================================================

def load_gdf_file(filepath):

    print(f"\nLoading: {filepath}")

    # --------------------------------------------------------
    # Read GDF
    # --------------------------------------------------------

    raw = mne.io.read_raw_gdf(
        filepath,
        preload=True,
        verbose=False
    )

    # --------------------------------------------------------
    # Find EEG channels
    # --------------------------------------------------------

    eeg_channels = find_eeg_channels(raw)

    print("EEG channels:", eeg_channels)

    raw.pick(eeg_channels)

    # --------------------------------------------------------
    # Resample to 250 Hz if necessary
    # --------------------------------------------------------

    if raw.info["sfreq"] != FS:

        print(
            f"Resampling "
            f"{raw.info['sfreq']} Hz -> {FS} Hz"
        )

        raw.resample(FS)

    # --------------------------------------------------------
    # Band-pass filtering
    # --------------------------------------------------------

    raw.filter(
        l_freq=LOW_FREQ,
        h_freq=HIGH_FREQ,
        verbose=False
    )

    data = raw.get_data()

    print(
        "Continuous EEG shape:",
        data.shape
    )

    # --------------------------------------------------------
    # Read annotations
    # --------------------------------------------------------

    events, event_dict = mne.events_from_annotations(
        raw,
        verbose=False
    )

    print(
        "Event dictionary:",
        event_dict
    )

    # --------------------------------------------------------
    # Locate 769 / 770
    #
    # MNE sometimes maps GDF annotation IDs to different
    # integer event IDs. Therefore we search event_dict
    # rather than assuming 769 and 770 are MNE IDs.
    # --------------------------------------------------------

    left_event = None
    right_event = None

    for name, event_id in event_dict.items():

        if str(name) == "769":
            left_event = event_id

        elif str(name) == "770":
            right_event = event_id

    if left_event is None or right_event is None:

        raise RuntimeError(
            "Could not find both motor-imagery events "
            "769 and 770."
        )

    print(
        f"Motor imagery IDs: "
        f"769 -> {left_event}, "
        f"770 -> {right_event}"
    )

    # --------------------------------------------------------
    # Extract trials
    # --------------------------------------------------------

    X = []
    y = []

    for sample, _, event_id in events:

        if event_id == left_event:

            start = int(sample)

            end = start + TRIAL_SAMPLES

            if end <= data.shape[1]:

                trial = data[
                    :,
                    start:end
                ]

                X.append(trial)
                y.append(0)

        elif event_id == right_event:

            start = int(sample)

            end = start + TRIAL_SAMPLES

            if end <= data.shape[1]:

                trial = data[
                    :,
                    start:end
                ]

                X.append(trial)
                y.append(1)

    if len(X) == 0:

        raise RuntimeError(
            f"No motor-imagery trials found in "
            f"{filepath}"
        )

    X = np.asarray(
        X,
        dtype=np.float32
    )

    y = np.asarray(
        y,
        dtype=np.int64
    )

    print(
        "Trials:",
        X.shape,
        "Labels:",
        y.shape
    )

    print(
        "Class distribution:",
        np.bincount(y)
    )

    return X, y


# ============================================================
# LOAD ONE SUBJECT
# ============================================================

def load_subject(
    subject,
    data_dir=DATA_DIR
):

    subject = int(subject)

    training_files = [
        f"B{subject:02d}01T.gdf",
        f"B{subject:02d}02T.gdf",
        f"B{subject:02d}03T.gdf"
    ]

    all_X = []
    all_y = []

    print("\n" + "=" * 60)
    print(
        f"LOADING SUBJECT {subject:02d}"
    )
    print("=" * 60)

    for filename in training_files:

        filepath = os.path.join(
            data_dir,
            filename
        )

        if not os.path.exists(filepath):

            print(
                f"WARNING: {filepath} not found"
            )

            continue

        X, y = load_gdf_file(
            filepath
        )

        all_X.append(X)
        all_y.append(y)

    if len(all_X) == 0:

        raise RuntimeError(
            f"No training files found "
            f"for subject {subject}"
        )

    X = np.concatenate(
        all_X,
        axis=0
    )

    y = np.concatenate(
        all_y,
        axis=0
    )

    # --------------------------------------------------------
    # Normalize per trial/channel
    # --------------------------------------------------------

    mean = X.mean(
        axis=2,
        keepdims=True
    )

    std = X.std(
        axis=2,
        keepdims=True
    )

    X = (
        X - mean
    ) / (
        std + 1e-6
    )

    print("\nSubject summary:")

    print(
        "X:",
        X.shape
    )

    print(
        "y:",
        y.shape
    )

    print(
        "classes:",
        np.bincount(y)
    )

    return X.astype(np.float32), y


# ============================================================
# LOAD MULTIPLE SUBJECTS
# ============================================================

def load_multiple_subjects(
    subjects,
    data_dir=DATA_DIR
):

    X_all = []
    y_all = []
    domain_all = []

    for domain, subject in enumerate(subjects):

        X, y = load_subject(
            subject,
            data_dir
        )

        domains = np.full(
            len(y),
            domain,
            dtype=np.int64
        )

        X_all.append(X)
        y_all.append(y)
        domain_all.append(domains)

    X_all = np.concatenate(
        X_all,
        axis=0
    )

    y_all = np.concatenate(
        y_all,
        axis=0
    )

    domain_all = np.concatenate(
        domain_all,
        axis=0
    )

    return (
        X_all,
        y_all,
        domain_all
    )


# ============================================================
# MAIN TEST
# ============================================================

if __name__ == "__main__":

    print("=" * 60)
    print("TESTING DATASET LOADER")
    print("=" * 60)

    X, y = load_subject(1)

    print("\nFINAL:")

    print(
        "X shape:",
        X.shape
    )

    print(
        "y shape:",
        y.shape
    )

    print(
        "Mean:",
        X.mean()
    )

    print(
        "Std:",
        X.std()
    )

    print(
        "\nDATASET LOADER TEST SUCCESSFUL"
    )
