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
      major_module_ablation_summary.csv
      major_module_targeted_ablation_table.csv
      all_paper_runs_summary.csv
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

The README keeps only the core equations needed to understand the model. The full manuscript contains the more detailed derivation and ablation definitions.

### 1.1 Problem Setup

For each subject-level waveform segment, the model receives synchronized PPG and ECG waveforms, with an optional numeric token when auxiliary channels are available. The target is joint SBP/DBP regression plus four-level BP risk stratification:

```math
\mathbf{x}_i = (\mathbf{x}^{\mathrm{ppg}}_i, \mathbf{x}^{\mathrm{ecg}}_i, \mathbf{x}^{\mathrm{num}}_i), \quad
\mathbf{y}_i = [y^{\mathrm{SBP}}_i, y^{\mathrm{DBP}}_i], \quad
c_i = f_{\mathrm{risk}}(\mathbf{y}_i).
```

The risk label $c_i$ corresponds to `Normal`, `Elevated`, `Stage 1`, or `Stage 2`. The experimental protocol is subject-disjoint: validation and test subjects are separated from training subjects, and held-out subjects are further split into support/query segments for few-shot personalization.

### 1.2 High-Level Architecture

SAQM-MedFuse combines five coupled methodological components:

1. QESF: Quality-aware Encoding and Sparse Fusion
2. URC: Uncertainty-aware Regression and Credibility Aggregation
3. ADSF: Anchor-guided Decision and Safe Posterior Fusion
4. STC: Safety-aware Tail Calibration and Residual Correction
5. POCR: Personalization, Objective design, and Conformal Reliability

The implementation is intentionally multi-stage. Earlier stages learn multimodal BP representations, while later stages select operating points, correct high-risk failure modes, calibrate subject-level predictions, and evaluate uncertainty reliability.

### 1.3 QESF: Quality-Aware Encoding and Sparse Fusion

For each modality $m \in \{\mathrm{ppg}, \mathrm{ecg}\}$, the model extracts a learned waveform representation $\mathbf{h}^{(m)}_i$ and a quality descriptor $\boldsymbol{\phi}^{(m)}_i$. The quality score and modality token are:

```math
q^{(m)}_i = \sigma(g^{(m)}_q(\boldsymbol{\phi}^{(m)}_i)), \quad
\mathbf{t}^{(m)}_i = [\mathbf{h}^{(m)}_i \Vert q^{(m)}_i \Vert a^{(m)}_i].
```

Here, $a^{(m)}_i$ is the modality availability indicator. A cross-modal descriptor $\mathbf{c}_i$ summarizes representation difference, element-wise interaction, quality scores, and modality availability. The router then selects a sparse mixture of PPG, ECG, joint, and cross-modal experts:

```math
\boldsymbol{\alpha}_i = \mathrm{TopKSoftmax}(r([\mathbf{t}^{\mathrm{ppg}}_i \Vert \mathbf{t}^{\mathrm{ecg}}_i \Vert \mathbf{t}^{\mathrm{num}}_i \Vert \mathbf{c}_i])), \quad
\mathbf{z}_i = \sum_{e=1}^{E}\alpha_{i,e}\mathbf{z}^{(e)}_i.
```

This design lets the model reduce reliance on corrupted or missing modalities rather than treating PPG and ECG as equally reliable for every sample.

### 1.4 URC: Uncertainty-Aware Regression and Credibility Aggregation

SAQM-MedFuse attaches regression branches to the PPG, ECG, and fused representations. Each branch predicts a BP mean and a log-variance vector:

```math
(\boldsymbol{\mu}^{(b)}_i,\mathbf{s}^{(b)}_i) = g^{(b)}_{\mathrm{reg}}(\mathbf{z}^{(b)}_i), \quad
\mathbf{s}^{(b)}_i = \log \boldsymbol{\sigma}^{2,(b)}_i.
```

A credibility head estimates branch weights $w^{(b)}_i$ from branch features, quality, availability, and uncertainty cues. The initial BP estimate and uncertainty proxy are:

```math
\hat{\mathbf{y}}^{(0)}_i = \sum_b w^{(b)}_i\boldsymbol{\mu}^{(b)}_i, \quad
u_i = \sum_b w^{(b)}_i\bar{\sigma}^{2,(b)}_i + \sum_b w^{(b)}_i\|\boldsymbol{\mu}^{(b)}_i-\hat{\mathbf{y}}^{(0)}_i\|_2^2/2.
```

The uncertainty proxy combines heteroscedastic branch variance and disagreement across branches. It is later reused for risk posterior estimation and safety-aware decision fusion.

### 1.5 ADSF: Anchor-Guided Decision and Safe Posterior Fusion

The decision layer uses complementary anchors rather than relying on a single posterior source. One anchor favors regression stability; the other favors class-boundary discrimination. A guided decision head is trained with classification, distillation, ordinal, and proxy-BP objectives:

```math
\mathcal{L}_{\mathrm{guide}} =
\mathcal{L}_{\mathrm{focal}} + \lambda_{\mathrm{KD}}\mathcal{L}_{\mathrm{KD}} +
\lambda_{\mathrm{ord}}\mathcal{L}_{\mathrm{ord}} + \lambda_{\mathrm{bp}}\mathcal{L}_{\mathrm{proxyBP}}.
```

The final classifier blends multiple posterior sources. When the decision posterior and the regression-induced posterior disagree in high-risk regions, a bounded safety gate pulls the final posterior toward the safer regression-derived evidence:

```math
\mathbf{p}^{\mathrm{safe}}_i = (1-\lambda_i)\tilde{\mathbf{p}}_i + \lambda_i\mathbf{p}^{\mathrm{reg}}_i.
```

Here, $\tilde{\mathbf{p}}_i$ is the selected decision posterior, $\mathbf{p}^{\mathrm{reg}}_i$ is the regression-induced posterior, and $\lambda_i$ is a bounded sample-wise safety weight.

### 1.6 STC: Safety-Aware Tail Calibration and Residual Correction

Cuffless BP models often shrink rare high-BP samples toward the population mean. SAQM-MedFuse therefore adds a safety-oriented correction stage for clinically undesirable upper-tail underestimation:

```math
\hat{\mathbf{y}}_i =
\hat{\mathbf{y}}^{(0)}_i + \boldsymbol{\rho}_i \odot \boldsymbol{\delta}^{\mathrm{mono}}_i + \boldsymbol{\delta}^{\mathrm{tail}}_i.
```

The monotone term $\boldsymbol{\delta}^{\mathrm{mono}}_i$ is controlled by high-risk evidence, while $\boldsymbol{\delta}^{\mathrm{tail}}_i$ uses upper-quantile proposals and a safety margin:

```math
\boldsymbol{\delta}^{\mathrm{tail}}_i =
\boldsymbol{\kappa}_i \odot \mathrm{ReLU}(\mathbf{q}^{\tau}_i - \hat{\mathbf{y}}^{(0)}_i - \mathbf{m}).
```

The operating point is selected on validation data under bounded-deterioration constraints, so upper-tail protection should not create unacceptable global MAE or conformal-coverage drift.

### 1.7 POCR: Personalization and Conformal Reliability

For each held-out subject, support segments estimate a subject-adaptive affine calibration:

```math
\tilde{\mathbf{y}}_i = \mathbf{s}_{u(i)} \odot \hat{\mathbf{y}}_i + \mathbf{b}_{u(i)}.
```

The scale and bias parameters are shrunk toward global priors when the subject support set is small. The training objective combines regression, uncertainty, auxiliary classification, router/credibility regularization, and tail-safety terms:

```math
\mathcal{L} = \mathcal{L}_{\mathrm{reg}} + \lambda_{\mathrm{NLL}}\mathcal{L}_{\mathrm{NLL}} + \lambda_{\mathrm{aux}}\mathcal{L}_{\mathrm{aux}} + \lambda_{\mathrm{router}}\mathcal{L}_{\mathrm{router}} + \lambda_{\mathrm{cred}}\mathcal{L}_{\mathrm{cred}} + \lambda_{\mathrm{tail}}\mathcal{L}_{\mathrm{tail}}.
```

Split conformal prediction is used to report interval reliability for SBP and DBP:

```math
\mathcal{I}^{(d)}_i = [\tilde{y}^{(d)}_i - q^{(d)}_{1-\alpha},\ \tilde{y}^{(d)}_i + q^{(d)}_{1-\alpha}].
```

Together, personalization and conformal reliability convert the global multimodal estimator into a subject-adaptive, uncertainty-aware decision-support pipeline.

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

The current release package includes the completed major-module `paperfix`
ablation summaries. For manuscript reporting, use
`major_module_targeted_ablation_table.csv` as the compact module-specific table
and `major_module_ablation_summary.csv` as the full audit table.

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
