# Multimodal Depression Detection from Acoustic and Linguistic Features

**Data Science Tools and Techniques — Term Project, FAST-NUCES Lahore**

This project asks whether **speech and language from a clinical interview can detect depression** (binary PHQ-8 ≥ 10), and whether a **compact, stable feature subset** can keep the performance of the full feature set. The output is a screening aid that flags participants for follow-up. It is not a diagnostic tool.

## Team

| Name | ID | Responsibilities |
|---|---|---|
| Hajra Abid | L26-8019 | EDA and data preparation (Part 1) |
| Ali Hussan | 25L-8007 | Feature extraction, modeling and fusion |
| H. Maliha Shabbir | 24L-9250 | Evaluation, stability analysis, prediction pipeline and integration |

## Dataset

- **DAIC-WOZ** (Distress Analysis Interview Corpus – Wizard of Oz), USC Institute for Creative Technologies, as released for the AVEC 2017 depression sub-challenge.
- **Licence-restricted.** It requires a signed data-use agreement with USC ICT. **The data is not included in this repository and must not be redistributed.**
- One semi-structured interview per participant with a virtual interviewer ("Ellie"). Each participant has audio, a transcript, COVAREP and formant acoustic features, and OpenFace facial tracking.
- **188 usable participants** (IDs 300–492). Participant 440 is in the official dev split but has no data.
- **Official splits:** train 107, dev 34, test 47.
- **Target:** official `PHQ8_Binary` (PHQ-8 ≥ 10). About 29 % of participants are depressed.

Expected data layout (point `DAIC_ROOT` to this folder):

```
DAIC-WOZ/
├── train_split_Depression_AVEC2017.csv
├── dev_split_Depression_AVEC2017.csv
├── test_split_Depression_AVEC2017.csv
├── full_test_split.csv
├── DAIC_WOZ_wav2vec2_temporal_1536.csv     # pre-computed speech embeddings (used by Part 2)
├── DAIC_WOZ_MentalBERT_1536.csv            # pre-computed text embeddings (used by Part 2)
├── DAIC_WOZ_linguistic_features_12.csv     # pre-computed linguistic statistics (validated in Part 1, not modeled)
├── DAIC_WOZ_video_temporal_348.csv         # pre-computed video features (validated in Part 1, not modeled)
└── 300_P/ … 492_P/   (or 300_P.zip … 492_P.zip — Part 1 extracts them)
    ├── XXX_AUDIO.wav, XXX_TRANSCRIPT.csv, XXX_COVAREP.csv, XXX_FORMANT.csv
    └── XXX_CLNF_AUs.txt, XXX_CLNF_pose.txt, XXX_CLNF_gaze.txt, XXX_CLNF_features(3D).txt
```

## Repository structure

```
.
├── README.md
├── requirements.txt
├── DSTT_Term_Project_Report.docx / .pdf    # project report
├── dstt-project-eda.ipynb                  # Part 1: acquisition → cleaning → preprocessing → features → EDA
├── dstt-project-part2-classifier.ipynb     # Part 2: models, fusion, compact subsets, stability, final test
├── part2_models.py                         # LateFusion estimator (written by Part 2)
├── predict.py                              # prediction for unseen participants (written by Part 2)
├── part1_outputs/
│   ├── features_master.csv                 # 188 participants × 191 hand-crafted features (+ split)
│   ├── qc_per_participant.csv, outlier_summary.csv, feature_ranking_devpool.csv, eda_summary.csv
│   ├── cache/participants/*.json           # per-participant extraction cache
│   └── figures/
└── part2_outputs/
    ├── final_model.joblib                  # saved model + threshold + expected columns
    ├── rq1_all_candidates.csv, rq2_fusion.csv, rq3_compact_subsets.csv, rq4_stability.csv
    ├── leaderboard_nested_cv.csv, test_results.csv, answers.txt
    ├── cache/                              # per-candidate nested-CV results
    └── figures/
```

## Setup

Python 3.10+ (developed on Python 3.12, Windows; also runs on Kaggle).

```bash
pip install -r requirements.txt
```

## How to run

Both notebooks detect Kaggle automatically: if `/kaggle/input/datasets/malihashabbir/depressiondata` exists they read from there and write to `/kaggle/working/`. Locally, set `DAIC_ROOT`:

```bash
# Windows (PowerShell)
$env:DAIC_ROOT = "D:/DAIC-WOZ"
jupyter notebook
```

```bash
# Or run top-to-bottom from the command line
DAIC_ROOT=D:/DAIC-WOZ python -m nbconvert --to notebook --execute --inplace dstt-project-eda.ipynb
DAIC_ROOT=D:/DAIC-WOZ python -m nbconvert --to notebook --execute --inplace dstt-project-part2-classifier.ipynb
```

| Notebook | First run | Re-run (cached) |
|---|---|---|
| Part 1 | ≈ 10 min with 10 CPU workers (≈ 1 min per participant per core; set `N_JOBS`) | < 1 min |
| Part 2 | ≈ 40 min on 12 cores (nested cross-validation over ~80 candidates); needs Part 1's `features_master.csv` | < 1 min |

Predict for new participants (feature files in the same format as the DAIC-WOZ tables):

```bash
python predict.py --features-dir D:/DAIC-WOZ part1_outputs --out predictions.csv
```

## Method summary

**Part 1 (data → features → EDA)**
- Read-only inventory, then validation of labels and splits: target rule, duplicates, split overlap, and files that are missing or unexpected.
- **Participant speech only:** interviewer turns are removed from the text, and the audio is cut to the participant's own turns.
- Features, one row per participant:
  - eGeMAPS v02 (openSMILE): pitch, loudness, jitter, shimmer, HNR, formants, spectral and voicing statistics
  - MFCC 1–13
  - COVAREP voice quality on voiced frames
  - formants F1–F3
  - turn timing: speaking time, speech rate, response latency
  - linguistic and lexicon features
  - supplementary OpenFace action units, pose and gaze, using successfully tracked frames only
- Cleaning:
  - missing values are summarised, then imputed inside training folds in Part 2
  - constant and near-duplicate features are removed
  - outliers are summarised and kept, with the reasons given
  - skewed features are log-transformed
- EDA and univariate effect sizes use the **development pool (train + dev) only**, with false-discovery-rate correction.

**Part 2 (modeling)**
- **Development pool = train + dev (141 participants).** Nested, stratified cross-validation (outer 5 folds × 2 repeats, inner 5 folds). Scaling, SMOTE, feature selection, hyper-parameters and the decision threshold are all fitted inside training folds.
- The official **test split (47 participants) is used once**, for two final models declared in advance.
- Feature blocks: **hand-crafted acoustic and text features from Part 1** (main, per the proposal), plus pre-computed **MentalBERT** text embeddings and **wav2vec2** speech embeddings (extension).
- Models: logistic regression, RBF-SVM, random forest and gradient boosting. Class imbalance is handled with class weighting or SMOTE.
- Fusion designs: early fusion, late fusion with equal weights, and weighted late fusion with a stacker.
- Compact subsets: ANOVA top-k and L1 logistic regression. Stability is measured with the Nogueira index and Jaccard similarity over 25 resamples.

## Results

Primary metric: macro-F1. Majority-class baseline: 0.415.

| Research question | Result (nested CV on 141 participants) |
|---|---|
| **RQ1** Beats majority baseline? | Yes. Acoustic: hand-crafted **0.572**, wav2vec2 **0.573**, both **0.613**. Text: hand-crafted **0.514**, MentalBERT **0.695**, both **0.697** |
| **RQ2** Fusion vs best single modality? | Early fusion of all speech + text blocks **0.732** (+0.036 over text, better in 6 of 10 folds): suggestive, not conclusive. Late fusion 0.619 (equal) and 0.639 (weighted); hand-crafted early fusion 0.569 |
| **RQ3** Compact subset within 2 points? | Yes. 100 of 3,217 features **0.712** vs 0.732 (200 features: 0.735). Hand-crafted: 10 of 145 features 0.564 vs 0.569 |
| **RQ4** Stable selection? | ANOVA filter moderately stable (Nogueira 0.52; 32 features chosen in ≥ 80 % of runs); L1 unstable on all features (0.30). The stable hand-crafted features match the Part 1 EDA |

**Optimization:** default model 0.701 → class weighting, threshold tuning, SMOTE and feature selection → **0.735**. Recall for the depressed class rose from 0.49 to 0.62.

**One-shot test (47 participants, 14 depressed):**

| Pre-declared model | CV macro-F1 | Test macro-F1 | Test AUC |
|---|---|---|---|
| Best overall: early fusion, logistic regression, top-200 features | 0.735 | 0.676 | 0.721 |
| Best compact: early fusion, logistic regression, top-100 features | 0.712 | 0.631 | 0.727 |

With 14 depressed test participants, one person moves macro-F1 by about 0.02–0.04.

**Key EDA findings (Part 1):**
- No single hand-crafted feature separates the classes: 14 of 191 features reach p < 0.05, against about 10 expected by chance, and none survive FDR correction.
- The largest effects are medium-sized (d ≈ 0.5):
  - more anxiety and negative-emotion words
  - breathier phonation (higher H1–H2)
  - more variable voice quality
  - steeper rising pitch slopes
- Depressed participants also pause somewhat longer before answering.

## Report

`DSTT_Term_Project_Report.docx` / `.pdf`: an 8-page report covering introduction, methodology, results, analysis and conclusion. Every number in it comes from the executed notebooks.

## Limitations

- Small sample (141 development, 47 test participants). Fold-to-fold standard deviation of macro-F1 is about 0.05–0.12.
- The wav2vec2 and MentalBERT embeddings were pre-computed outside these notebooks. We assume they were extracted from participant speech only.
- PHQ-8 is a self-report questionnaire, so the labels themselves are noisy. Participant 409 (score 10) carries an official label of 0, and the official labels are kept for comparability with published work.
- Model performance by gender was unstable: better for women on the development pool, better for men on the test set. Monitor subgroups before any real use.
