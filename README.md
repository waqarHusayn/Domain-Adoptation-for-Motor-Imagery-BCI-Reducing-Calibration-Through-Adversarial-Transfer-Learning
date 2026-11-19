# Domain Adaptation for Motor Imagery BCI

Cross subject motor imagery EEG classification with an EEGNet style network and domain adversarial training, evaluated with leave one subject out (LOSO) on BCI Competition IV 2b.

Final Year Design Project (FYDP), Department of Computer Systems Engineering, Sukkur IBA University.

**Status:** first stage complete (working prototype with LOSO results). Calibration budget experiments, artifact cleaning and the lambda ablation are in progress. See [Project status](#project-status).

## Overview

A motor imagery brain computer interface (BCI) lets a user control a device by imagining a movement. A model trained on one person usually fails on another because EEG differs between people, so every new user needs a long calibration session (often 1 to 2 hours).

This project studies whether domain adversarial learning can transfer knowledge across subjects and reduce that calibration. A shared feature extractor feeds two heads: a class head (left or right hand) and a domain head connected through a gradient reversal layer, which pushes the extractor toward features that hide the subject identity.

## Results so far

Leave one subject out over 9 subjects, single run (seed 42). Class 1 (right hand) is the positive class for precision, recall and F1.

| Metric | Mean | Std. dev. |
|---|---|---|
| Accuracy | 73.16% | 7.38 pp |
| Precision | 73.80% | 8.53 pp |
| Recall | 72.74% | 11.04 pp |
| F1 | 72.86% | 8.02 pp |
| Cohen's kappa | 0.463 | 0.148 |

| Method | Mean accuracy | Mean kappa |
|---|---|---|
| **Adversarial EEGNet style model (ours)** | **73.16%** | **0.463** |
| LDA + band power | 66.41% | 0.328 |
| Logistic regression + band power | 66.17% | 0.323 |
| Linear SVM + band power | 66.10% | 0.322 |
| RBF SVM + band power | 64.17% | 0.283 |
| CSP + LDA | 63.91% | 0.278 |
| Random forest + band power | 62.25% | 0.245 |

Per subject accuracy: S1 78.50, S2 63.25, S3 61.75, S4 86.43, S5 74.29, S6 73.50, S7 74.25, S8 72.73, S9 73.75.

How to read these results:
* The model uses **unlabeled** target trials in training (through the domain head). Target class labels are never used in training. This is the zero labeled target trials case.
* The baselines use no target data, and their preprocessing differs slightly, so the comparison is indicative, not fully controlled.
* The 82% accuracy in some result files comes from a pooled 50 trial holdout where subjects appear in both training and testing. It is **not** subject independent. Use the LOSO numbers.

## Method

**Data.** BCI Competition IV 2b: 9 subjects, left vs right hand motor imagery. Only the three labeled training sessions per subject are used (`Bxx01T`, `Bxx02T`, `Bxx03T`), giving 3,680 trials (1,840 per class). Channels C3, Cz, C4.

**Preprocessing.**
* Sampling rate 250 Hz
* Band pass 4 to 40 Hz (FIR, Hamming window, zero phase, MNE defaults)
* Events 769 (left, class 0) and 770 (right, class 1)
* Epoch: 0 to 4 s from the cue, 1,000 samples per channel
* Per trial and per channel normalization: `(x - mean) / (std + 1e-6)`
* Model input shape `[batch, 1, 3, 1000]`

**Model.** Temporal convolution (16 filters, 1 x 64), depthwise spatial convolution (3 x 1), average pooling, separable convolution, flatten to 992 features. Two heads (992 to 64 to 2): class head and domain head. 130,180 trainable parameters.

**Objective.**

```
L_cls = -(1/Ns) * sum log p(y_i | x_i)
L_dom = average of source (d=0) and target (d=1) cross entropy
L     = L_cls + w * L_dom,   w = 0.1
GRL   : forward R(x) = x,   backward dR/dx = -lambda,   lambda = 1.0
```

**Training.** Adam, learning rate 0.001, weight decay 0.0001, batch size 32, 30 epochs, gradient clipping at 5, seed 42.

**Baselines.** LDA, logistic regression, linear SVM, RBF SVM, random forest on log band power (Welch, 6 bands, 3 channels = 18 features), and CSP (4 components) + LDA.

## Project structure

The repository currently has a flat layout. A cleaner `src/` layout is planned.

```
README.md
requirements.txt          (to be completed)
dataset.py                MNE loading, filtering, epoching, normalization
model.py                  EEGNet style extractor, class head, GRL, domain head
train.py                  single target workflow (subject 1 as target)
evaluate.py               evaluates the single target checkpoint
loso_adversarial.py       LOSO training and evaluation, all 9 subjects
train_test.py             50 trial pooled holdout workflow
baselines.py              classical baselines with LOSO
plot_results.py           plots from the LOSO results CSV
initialdataset.py         earlier pipeline kept for reference
check_dataset.py          placeholder (to be implemented)
results/                  LOSO metrics, summary CSV, plots
results_baselines/        baseline metrics and comparison plots
docs/                     technical guide, paper draft, slides
```

## Setup

Requires Python 3.10 or newer.

```bash
git clone https://github.com/<username>/<repository>.git
cd <repository>

python -m venv .venv
# Windows:  .venv\Scripts\activate
# Linux/Mac: source .venv/bin/activate

pip install numpy scipy pandas matplotlib scikit-learn mne torch
```

Once `requirements.txt` is completed, use `pip install -r requirements.txt` instead.

### Get the data

The dataset is not included in this repository.

1. Download BCI Competition IV, dataset 2b (GDF files) from the competition page: https://www.bbci.de/competition/iv/
2. Place the 45 `.gdf` files in a folder named `BCICIV_2b_gdf/` in the project root.
3. Only the 27 training files (`Bxx01T`, `Bxx02T`, `Bxx03T`) are used by the loaders.

Several scripts currently contain hard coded Windows paths (notably `baselines.py` and `train_test.py`). Change the data path at the top of each script before running.

## Usage

```bash
python loso_adversarial.py   # LOSO for subjects 1 to 9, saves checkpoints and loso_results.csv
python baselines.py          # six classical baselines with LOSO
python plot_results.py       # accuracy, F1 and kappa plots
python train_test.py         # pooled 50 trial holdout run (not subject independent)
```

## Project status

Done:
- [x] Preprocessing pipeline (4 to 40 Hz, 4 second epochs, normalization)
- [x] Adversarial EEGNet style model
- [x] LOSO evaluation over 9 subjects
- [x] Six classical baselines

In progress or planned:
- [ ] Artifact handling: EOG regression for blinks and trial rejection, then rerun LOSO with and without cleaning
- [ ] Calibration budgets of 0, 10, 20, 50 and 100 labeled target trials (last layer fine tuning)
- [ ] Extra baselines: train from scratch, no adaptation, fine tuning
- [ ] Lambda ablation over 0, 0.01, 0.05, 0.1, 0.5, 1.0
- [ ] Multiple seeds, confidence intervals and paired significance tests
- [ ] Confusion matrices and t-SNE plots
- [ ] BCI Competition IV 2a (22 channels, 4 classes) as an extension
- [ ] Repository cleanup: pinned requirements, no hard coded paths, one canonical pipeline, smoke tests

## Known limitations

* Single run (one seed) and nine subjects, so statistical power is limited.
* Three preprocessing variants exist across the scripts. One canonical pipeline will be chosen.
* The LOSO script selects its model by training loss and has no validation set.
* Healthy subject data only. No clinical claim is made.

## Contributing

We work with branches and pull requests.


## Team
|---|---|
| Waqar Hussain| @waqarHusayn |
| Manav Dev | @manav_dev468 |
| Junaid Ahmad | @junaid-20 |

Supervisor: Dr. Mumtaz Kaloi

## References

1. Lawhern, V.J. et al. EEGNet: A compact convolutional neural network for EEG-based brain-computer interfaces. *J. Neural Eng.* 2018, 15, 056013.
2. Ganin, Y. et al. Domain-Adversarial Training of Neural Networks. *J. Mach. Learn. Res.* 2016, 17, 1-35.
3. Tangermann, M. et al. Review of the BCI Competition IV. *Front. Neurosci.* 2012, 6, 55.

## License

To be decided with the supervisor. The BCI Competition IV data has its own terms of use and is not redistributed here.
