# SAQM-MedFuse Codes and Reproducibility Package

This repository provides the reproducibility code and aggregate paper results for:

> SAQM-MedFuse: Safety-aware Quality-driven Multimodal Fusion for Cuffless Blood Pressure Estimation and Risk Stratification

SAQM-MedFuse is a research-stage multimodal framework for joint systolic/diastolic blood pressure (SBP/DBP) estimation and four-level blood pressure risk stratification from synchronized photoplethysmography (PPG) and electrocardiography (ECG). The code is organized to reproduce the main experiment, representative comparison baselines, and the paper-facing major-module ablation suite.

Important: this repository does not include raw MIMIC-BP, MIMIC-III, or PhysioNet data. Users must download the dataset independently and comply with the corresponding data license and use agreement.

## Repository Layout

```text
SAQM-MedFuse/
  README.md
  LICENSE
  CITATION.cff
  requirements.txt
  configs/
    paths.example.ps1
    paths.example.yaml
  scripts/
    run_v10_18_main.ps1
    run_v10_18_comparison_experiments.ps1
    run_v10_18_major_module_ablation_experiments.ps1
  docs/
    UPLOAD_GUIDE.md
  results/
    paper_summary/
      main_version_summary.csv
      comparison_summary.csv
      lite_train_ablation_summary.csv
      fast_ablation_summary.csv
      ...
    manifests/
      major_module_ablation_manifest.json
      ...
    figures/
      selected paper figures
  *.py
    core model, training, comparison, ablation, and summary scripts
```

The `results/` directory contains aggregate, paper-level outputs only. It intentionally excludes raw waveforms, per-subject predictions, model checkpoints, and private local paths.

## 1. Model Explanation

### 1.1 Problem Setup

For each subject-level waveform segment, the model receives synchronized:

- PPG waveform: `x_ppg`
- ECG waveform: `x_ecg`
- optional numeric token: set to zero when auxiliary numeric channels are not available

The regression target is:

```text
y = [SBP, DBP]
```

The auxiliary risk stratification target uses four clinical BP categories:

```text
Normal, Elevated, Stage 1, Stage 2
```

The experimental protocol is subject-disjoint. In the paper setting, subjects are divided into train, validation, and test groups. Validation and test subjects are further split into support/query segments so that few-shot subject personalization is evaluated without subject leakage.

### 1.2 High-Level Architecture

SAQM-MedFuse combines five coupled methodological components:

1. QESF: Quality-aware Encoding and Sparse Fusion
2. URC: Uncertainty-aware Regression and Credibility Aggregation
3. ADSF: Anchor-guided Decision and Safe Posterior Fusion
4. STC: Safety-aware Tail Calibration and Residual Correction
5. POCR: Personalization, Objective design, and Conformal Reliability

The implementation is intentionally multi-stage. Earlier stages learn multimodal BP representations, while later stages select operating points, correct high-risk failure modes, calibrate subject-level predictions, and evaluate uncertainty reliability.

### 1.3 QESF: Quality-Aware Encoding and Sparse Fusion

For each modality, the model extracts waveform features and a quality descriptor. The quality descriptor summarizes morphology stability, local variability, spectral concentration, clipping behavior, outlier ratio, and peak-consistency statistics. These quality scores are used with modality availability indicators to build modality tokens.

The sparse fusion router activates a small set of experts:

- PPG expert
- ECG expert
- joint expert
- cross-modal expert

The router forms a fused representation through top-k quality-conditioned expert weighting. This design allows the model to reduce reliance on corrupted or missing modalities instead of assuming uniform modality reliability.

### 1.4 URC: Uncertainty-Aware Regression and Credibility Aggregation

SAQM-MedFuse attaches regression branches to the PPG, ECG, and fused representations. Each branch predicts both a BP mean and heteroscedastic uncertainty. A credibility head then combines branch features, quality scores, availability indicators, branch uncertainty, and routing statistics to estimate sample-wise branch weights.

The initial BP estimate is a credibility-weighted combination of branch predictions. Predictive uncertainty is represented by combining branch variance and cross-branch disagreement. This uncertainty is also used to derive a regression-induced BP risk posterior from clinical thresholds.

### 1.5 ADSF: Anchor-Guided Decision and Safe Posterior Fusion

The decision layer uses complementary anchors rather than a single posterior source:

- a calibration-oriented anchor, which favors regression stability
- a stage-sensitive anchor, which favors class-boundary discrimination

A guided decision head learns from anchor predictions, anchor disagreement, routing behavior, quality scores, and uncertainty descriptors. It is trained with classification, knowledge-distillation, ordinal-consistency, and proxy-BP objectives.

The final decision posterior is selected by validation-driven posterior fusion. A regression-aware safety fusion layer is activated when decision-level and regression-induced posteriors disagree in clinically relevant high-risk regions.

### 1.6 STC: Safety-Aware Tail Calibration and Residual Correction

Cuffless BP models often underestimate rare high-BP samples because the data distribution is dominated by central BP ranges. SAQM-MedFuse therefore includes safety-oriented upper-tail correction:

- high-risk guarding
- monotone high-bias calibration
- crisis-tail residual correction

The correction stage is selected under bounded-deterioration constraints so that upper-tail bias reduction does not cause unacceptable global MAE or reliability degradation.

### 1.7 POCR: Personalization and Conformal Reliability

For each held-out subject, support segments are used to estimate a subject-adaptive affine calibration:

```text
y_personalized = scale_subject * y_pred + bias_subject
```

The subject parameters are shrunk toward global estimates to avoid overfitting when the support set is small. Split conformal prediction is then used to report interval reliability for SBP and DBP.

## 2. Dataset Explanation

### 2.1 Dataset Used

The experiments use MIMIC-BP, a curated dataset for cuffless blood pressure estimation. The MIMIC-BP paper describes a derivative dataset with 380 hours of signals, including ABP, PPG, and ECG, from 1,524 anonymized subjects. Each subject has 30 waveform segments of 30 seconds.

Primary dataset citation:

```text
Sanches, I., Gomes, V.V., Caetano, C. et al.
MIMIC-BP: A curated dataset for blood pressure estimation.
Scientific Data 11, 1233 (2024).
https://doi.org/10.1038/s41597-024-04041-1
```

Dataset DOI:

```text
Samsung R&D Institute Brazil, SRBR et al. MIMIC-BP dataset.
https://doi.org/10.7910/DVN/DBM1NF
```

The MIMIC-BP article states that the dataset includes files such as `{abp,ppg,ecg,resp}.zip`, `labels.zip`, reading utilities, and subject split files. This code primarily requires PPG, ECG, and SBP/DBP labels; ABP is used by the dataset authors to derive the BP labels and can be retained locally for auditing.

### 2.2 Data License and Redistribution

Do not upload your local raw data directory, for example `<YOUR_LOCAL_MIMIC_BP_ROOT>`.

Do not upload extracted PPG/ECG/ABP/RESP arrays, labels, subject-level split files, per-subject predictions, or model checkpoints derived from restricted clinical data unless you have confirmed that redistribution is permitted under the relevant data license.

PhysioNet credentialed data rules prohibit sharing access to restricted data with third parties, and MIMIC-derived datasets or models should be handled under the same sensitivity assumptions as the source data. See:

- PhysioNet Credentialed Health Data License: https://physionet.org/content/mimiciv/view-license/3.1/
- MIMIC-IV derived dataset/model guidance: https://physionet.org/content/mimiciv/2.2/

### 2.3 Expected Local Data Structure

After downloading and extracting the MIMIC-BP files, arrange them locally as:

```text
<YOUR_LOCAL_MIMIC_BP_ROOT>/
  ppg/
    <subject_id>_ppg.npy
    or <subject_id>.npy
  ecg/
    <subject_id>_ecg.npy
    or <subject_id>.npy
  labels/
    <subject_id>_labels.npy
    or <subject_id>_label.npy
    or <subject_id>.npy
  splits/                 # optional
    train_subjects.txt
    val_subjects.txt
    calib_subjects.txt
    test_subjects.txt
```

The loader accepts common MIMIC-BP NPY layouts. Each subject should have readable PPG, ECG, and label files. Labels should contain SBP and DBP pairs.

## 3. Installation

### 3.1 Clone

```powershell
git clone https://github.com/yazyeah/SAQM-MedFuse.git
cd SAQM-MedFuse
```

### 3.2 Create Environment

Python 3.10 or 3.11 is recommended.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Install a PyTorch build matching your CUDA version if the default command does not install GPU-enabled PyTorch. Verify GPU availability:

```powershell
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"
```

### 3.3 Configure Local Paths

Copy the example path file:

```powershell
Copy-Item .\configs\paths.example.ps1 .\configs\paths.local.ps1
```

Edit `configs/paths.local.ps1`:

```powershell
$env:AQM_MIMIC_BP_ROOT = "<YOUR_LOCAL_MIMIC_BP_ROOT>"
$env:AQM_OUTPUT_ROOT = ""
$env:AQM_DEVICE = "cuda"
```

Then load it:

```powershell
. .\configs\paths.local.ps1
```

`configs/paths.local.ps1` is ignored by Git and should not be committed.

## 4. Reproduce Experiments

### 4.1 Main SAQM-MedFuse Result

```powershell
.\scripts\run_v10_18_main.ps1 `
  -Python .\.venv\Scripts\python.exe `
  -DataRoot $env:AQM_MIMIC_BP_ROOT `
  -Device cuda
```

Main output directory:

```text
outputs/mimic_bp_reg_v10_18_subjectdisjoint_baselineguard_crisisrepair_proto/
```

Main summary:

```text
outputs/v10_18_paper_summary/main_version_summary.csv
```

### 4.2 Representative Comparison Baselines

Run the main protocol first because the baselines reuse the same subject-disjoint split and calibration setting.

```powershell
.\scripts\run_v10_18_comparison_experiments.ps1 `
  -Python .\.venv\Scripts\python.exe `
  -DataRoot $env:AQM_MIMIC_BP_ROOT `
  -Device cuda `
  -Epochs 64 `
  -Patience 12
```

Default baselines:

- ANN-LSTM with ECG and PPG
- PPG BiLSTM with attention
- BP-Net CNN
- MLP-BP Mixer

Heavy optional baselines:

```powershell
.\scripts\run_v10_18_comparison_experiments.ps1 `
  -Python .\.venv\Scripts\python.exe `
  -DataRoot $env:AQM_MIMIC_BP_ROOT `
  -IncludeHeavy
```

Comparison summary:

```text
outputs/v10_18_paper_summary/comparison_summary.csv
```

### 4.3 Major-Module Ablation Suite

The paper-facing ablation suite removes the major conceptual modules:

- no quality/sparse fusion
- no uncertainty/credibility aggregation
- no anchor decision refinement
- no tail safety calibration
- no personalized conformal layer

Recommended practical command:

```powershell
.\scripts\run_v10_18_major_module_ablation_experiments.ps1 `
  -Python .\.venv\Scripts\python.exe `
  -DataRoot $env:AQM_MIMIC_BP_ROOT `
  -RunTag paperfix `
  -HeadEpochs 24 `
  -HeadPatience 6 `
  -HeadMinEpochs 8
```

This setting is designed to be reproducible within a realistic runtime. It reuses the full reference guided head for downstream-only ablations, while retraining the head for the quality/sparse-fusion proxy. Use the following only if you explicitly want fully independent variant heads:

```powershell
.\scripts\run_v10_18_major_module_ablation_experiments.ps1 `
  -Python .\.venv\Scripts\python.exe `
  -DataRoot $env:AQM_MIMIC_BP_ROOT `
  -RunTag independent_heads `
  -ForceTrainEachVariant
```

Run a single ablation variant:

```powershell
.\scripts\run_v10_18_major_module_ablation_experiments.ps1 `
  -Python .\.venv\Scripts\python.exe `
  -DataRoot $env:AQM_MIMIC_BP_ROOT `
  -Variant no_tail_safety_calibration `
  -RunTag paperfix
```

Ablation summary:

```text
outputs/v10_18_paper_summary/major_module_ablation_summary.csv
```

### 4.4 Refresh Paper Tables

```powershell
python -B .\summarize_v10_18_paper_runs.py
```

Combined paper summary:

```text
outputs/v10_18_paper_summary/all_paper_runs_summary.csv
```

Before adding a combined summary to GitHub, confirm that every row in the
corresponding experiment table is complete:

```powershell
.\scripts\check_experiment_completion.ps1 -SummaryDir .\results\paper_summary
```

## 5. Included Results

This repository includes aggregate results under:

```text
results/paper_summary/
results/manifests/
results/figures/
```

These files are intended for paper-level auditing and quick comparison. They are not a substitute for rerunning the experiments on the local dataset.

The included result files should be safe to publish because they are aggregate CSV/JSON summaries and selected figures. They should not contain raw waveforms, subject-level private files, model checkpoints, or per-sample prediction tables.

Note: the current release package excludes incomplete major-module `paperfix`
summary CSVs. After the major-module suite finishes, copy the completed
`major_module_ablation_summary.csv`, `major_module_ablation_manifest.json`, and
the refreshed `all_paper_runs_summary.csv` into `results/` only if all
`completed` flags are true.

## 6. Runtime Notes

The full main protocol and exhaustive search variants can be time-consuming. For practical reproduction:

- run the main v10.18 script first
- run comparison baselines after the main split exists
- run the major-module ablation suite with `-RunTag paperfix`
- avoid `-ForceTrainEachVariant` unless runtime is not a constraint
- use `-Variant <name>` to resume or debug one ablation at a time

Generated results are written to `outputs/`, which is ignored by Git.

## 9. Citation

If you use this repository, please cite the associated SAQM-MedFuse paper and the dataset:

```bibtex
@article{zhou2026saqmmedfuse,
  title   = {SAQM-MedFuse: Safety-aware Quality-driven Multimodal Fusion for Cuffless Blood Pressure Estimation and Risk Stratification},
  author  = {Zhou, Yuang and Li, Aoxuan and Qin, Junyuan},
  year    = {2026},
  journal = {Preprint}
}
```

```bibtex
@article{sanches2024mimicbp,
  title   = {MIMIC-BP: A curated dataset for blood pressure estimation},
  author  = {Sanches, Ivandro and Gomes, Victor V. and Caetano, Carlos and others},
  journal = {Scientific Data},
  volume  = {11},
  pages   = {1233},
  year    = {2024},
  doi     = {10.1038/s41597-024-04041-1}
}
```

```bibtex
@dataset{mimicbp_dataset_2023,
  title     = {MIMIC-BP dataset},
  author    = {{Samsung R&D Institute Brazil, SRBR} and others},
  year      = {2023},
  publisher = {Harvard Dataverse},
  doi       = {10.7910/DVN/DBM1NF}
}
```

## 10. Disclaimer

SAQM-MedFuse is provided for research reproducibility. It is not a medical device and must not be used as a stand-alone screening, diagnosis, or treatment system.
