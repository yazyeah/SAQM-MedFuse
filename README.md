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

For sample $i$, the input can be written as:

$$
\mathbf{x}_i = \left(\mathbf{x}^{ppg}_i, \mathbf{x}^{ecg}_i, \mathbf{x}^{num}_i\right),
$$

where $\mathbf{x}^{num}_i$ is optional. The regression target is:

$$
\mathbf{y}_i = [y^{sbp}_i, y^{dbp}_i] \in \mathbb{R}^2.
$$

The auxiliary risk stratification target uses four clinical BP categories:

```text
Normal, Elevated, Stage 1, Stage 2
```

The class label $c_i$ is derived from SBP/DBP clinical thresholds:

$$
c_i = f_{\mathrm{risk}}\left(y^{sbp}_i, y^{dbp}_i\right), \quad c_i \in \{0,1,2,3\}.
$$

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

For modality $m \in \{ppg, ecg\}$, let $\boldsymbol{\phi}^{(m)}_i$ denote its handcrafted quality descriptor and $\mathbf{h}^{(m)}_i$ denote its learned waveform representation. A lightweight quality predictor estimates:

$$
q^{(m)}_i = \sigma\left(g^{(m)}_q\left(\boldsymbol{\phi}^{(m)}_i\right)\right),
\quad q^{(m)}_i \in [0,1],
$$

where $\sigma(\cdot)$ is the sigmoid function. The modality token is:

$$
\mathbf{t}^{(m)}_i =
\left[
\mathbf{h}^{(m)}_i
\Vert q^{(m)}_i
\Vert a^{(m)}_i
\right],
$$

where $a^{(m)}_i$ is the modality availability indicator and $\Vert$ denotes concatenation.

The cross-modal descriptor is:

$$
\mathbf{c}_i =
\left[
\left|\mathbf{h}^{ppg}_i - \mathbf{h}^{ecg}_i\right|
\Vert
\mathbf{h}^{ppg}_i \odot \mathbf{h}^{ecg}_i
\Vert q^{ppg}_i
\Vert q^{ecg}_i
\Vert a^{ppg}_i
\Vert a^{ecg}_i
\right],
$$

where $\odot$ is element-wise multiplication. The sparse fusion router activates a small set of experts:

- PPG expert
- ECG expert
- joint expert
- cross-modal expert

The routing weights are computed by:

$$
\boldsymbol{\alpha}_i =
\mathrm{TopKSoftmax}
\left(
r\left(
\left[
\mathbf{t}^{ppg}_i
\Vert \mathbf{t}^{ecg}_i
\Vert \mathbf{t}^{num}_i
\Vert \mathbf{c}_i
\right]
\right)
\right).
$$

The fused representation is:

$$
\mathbf{z}_i =
\sum_{e=1}^{E} \alpha_{i,e}\mathbf{z}^{(e)}_i.
$$

This design allows the model to reduce reliance on corrupted or missing modalities instead of assuming uniform modality reliability. In the code, the corresponding architecture and quality-aware augmentation utilities are mainly implemented in `aqm_bp_shared_v9.py` and reused by later v10/v11 protocol scripts.

### 1.4 URC: Uncertainty-Aware Regression and Credibility Aggregation

SAQM-MedFuse attaches regression branches to the PPG, ECG, and fused representations. Each branch predicts both a BP mean and heteroscedastic uncertainty. A credibility head then combines branch features, quality scores, availability indicators, branch uncertainty, and routing statistics to estimate sample-wise branch weights.

For branch $b \in \{ppg, ecg, fuse\}$, the branch regressor outputs a mean prediction and a log-variance vector:

$$
\left(\boldsymbol{\mu}^{(b)}_i, \mathbf{s}^{(b)}_i\right)
=
g^{(b)}_{\mathrm{reg}}\left(\mathbf{z}^{(b)}_i\right),
\quad
\mathbf{s}^{(b)}_i = \log \boldsymbol{\sigma}^{2,(b)}_i.
$$

The credibility head predicts normalized branch weights:

$$
w^{(b)}_i =
\mathrm{softmax}_b
\left(
g_{\mathrm{cred}}
\left(
\mathbf{z}^{(b)}_i, q^{(b)}_i, a^{(b)}_i,
\boldsymbol{\sigma}^{2,(b)}_i
\right)
\right),
\quad
\sum_b w^{(b)}_i = 1.
$$

The initial BP estimate is:

$$
\hat{\mathbf{y}}^{(0)}_i =
\sum_b w^{(b)}_i \boldsymbol{\mu}^{(b)}_i.
$$

Predictive uncertainty is represented by combining heteroscedastic branch variance and cross-branch disagreement:

$$
u_i =
\sum_b w^{(b)}_i \bar{\sigma}^{2,(b)}_i
+
\sum_b w^{(b)}_i
\frac{
\left\|
\boldsymbol{\mu}^{(b)}_i - \hat{\mathbf{y}}^{(0)}_i
\right\|_2^2
}{2},
$$

where:

$$
\bar{\sigma}^{2,(b)}_i =
\frac{1}{2}
\sum_{d \in \{sbp, dbp\}}
\sigma^{2,(b)}_{i,d}.
$$

This uncertainty is also used to derive a regression-induced risk posterior. For example, the normal-class probability can be approximated by:

$$
p^{reg}_{i,\mathrm{normal}} =
\Phi\left(
\frac{120 - \hat{y}^{(0)}_{i,sbp}}{\sigma_{i,sbp}}
\right)
\Phi\left(
\frac{80 - \hat{y}^{(0)}_{i,dbp}}{\sigma_{i,dbp}}
\right),
$$

where $\Phi(\cdot)$ is the standard normal CDF. The final regression-induced posterior blends threshold-based and center-based posteriors:

$$
\mathbf{p}^{reg}_i =
\lambda_{\mathrm{th}}\mathbf{p}^{th}_i
+
\left(1-\lambda_{\mathrm{th}}\right)\mathbf{p}^{center}_i.
$$

### 1.5 ADSF: Anchor-Guided Decision and Safe Posterior Fusion

The decision layer uses complementary anchors rather than a single posterior source:

- a calibration-oriented anchor, which favors regression stability
- a stage-sensitive anchor, which favors class-boundary discrimination

Let the two anchor systems produce:

$$
\left(\hat{\mathbf{y}}^A_i, \mathbf{p}^A_i\right),
\quad
\left(\hat{\mathbf{y}}^B_i, \mathbf{p}^B_i\right).
$$

A guided decision head learns from anchor predictions, anchor disagreement, routing behavior, quality scores, and uncertainty descriptors. It is trained with classification, knowledge-distillation, ordinal-consistency, and proxy-BP objectives:

$$
\mathcal{L}_{guide}
=
\mathcal{L}_{focal}
+
\lambda_{KD}\mathcal{L}_{KD}
+
\lambda_{ord}\mathcal{L}_{ord}
+
\lambda_{bp}\mathcal{L}_{proxyBP}.
$$

The posterior mixture-of-experts combines anchor-derived posteriors, guided-head posteriors, and uncertainty-aware meta posteriors:

$$
\hat{\mathbf{p}}_i =
\sum_e \beta_{i,e}\mathbf{p}^{(e)}_i,
\quad
\sum_e \beta_{i,e}=1.
$$

To adapt the decision boundary to class imbalance, class reweighting is applied:

$$
\tilde{p}_i(c) =
\frac{
\hat{p}_i(c)^\gamma \nu_c
}{
\sum_{c'} \hat{p}_i(c')^\gamma \nu_{c'}
},
$$

where $\gamma$ controls posterior sharpness and $\nu_c$ is a class-specific reweighting factor.

A regression-aware safety fusion layer is activated when the decision posterior and regression-induced posterior disagree in clinically relevant high-risk regions:

$$
\mathbf{p}^{safe}_i =
\left(1-\lambda_i\right)\tilde{\mathbf{p}}_i
+
\lambda_i\mathbf{p}^{reg}_i,
$$

with:

$$
\lambda_i =
\mathrm{clip}
\left(
g_{safe}\left(\mathbf{d}_i, \mathbf{u}_i\right),
0,
\lambda_{\max}
\right).
$$

Here, $\mathbf{d}_i$ summarizes posterior disagreement and $\mathbf{u}_i$ summarizes uncertainty and upper-tail risk cues. The v10.18 decision search and safety fusion logic is implemented in `train_aqm_medfuse_mimic_bp_reg_v10_18_subjectdisjoint_baselineguard_crisisrepair_protocol.py`.

### 1.6 STC: Safety-Aware Tail Calibration and Residual Correction

Cuffless BP models often underestimate rare high-BP samples because the data distribution is dominated by central BP ranges. SAQM-MedFuse therefore includes safety-oriented upper-tail correction:

- high-risk guarding
- monotone high-bias calibration
- crisis-tail residual correction

The corrected BP estimate is written as:

$$
\hat{\mathbf{y}}_i =
\hat{\mathbf{y}}^{(0)}_i
+
\boldsymbol{\rho}_i \odot \boldsymbol{\delta}^{mono}_i
+
\boldsymbol{\delta}^{tail}_i,
$$

where $\boldsymbol{\rho}_i$ is a confidence gate, $\boldsymbol{\delta}^{mono}_i$ is a monotone high-risk calibration shift, and $\boldsymbol{\delta}^{tail}_i$ is a crisis-tail residual correction.

The crisis-tail correction uses high quantile evidence from branchwise or expertwise proposals:

$$
\boldsymbol{\delta}^{tail}_i =
\boldsymbol{\kappa}_i \odot
\mathrm{ReLU}
\left(
\mathbf{q}^{\tau}_i
-
\hat{\mathbf{y}}^{(0)}_i
-
\mathbf{m}
\right),
$$

where $\mathbf{q}^{\tau}_i$ is an upper quantile proposal, $\mathbf{m}$ is a safety margin, and $\boldsymbol{\kappa}_i$ is a gain vector controlled by crisis risk and uncertainty. The correction is intentionally nonnegative in the upper-tail setting so that strong high-risk evidence can increase, but not reverse, the safety shift.

The correction stage is selected under bounded-deterioration constraints:

$$
\theta^\star =
\arg\min_{\theta \in \mathcal{H}}
\mathcal{B}_{tail}(\theta),
$$

where:

$$
\mathcal{H}
=
\left\{
\theta:
\Delta MAE(\theta) \le \epsilon_{mae},
\quad
\Delta Gap_{cov}(\theta) \le \epsilon_{cov}
\right\}.
$$

This means the selected safety configuration must improve upper-tail behavior without causing unacceptable global error or conformal coverage drift.

### 1.7 POCR: Personalization and Conformal Reliability

For each held-out subject, support segments are used to estimate a subject-adaptive affine calibration:

$$
\tilde{\mathbf{y}}_i =
\mathbf{s}_{u(i)} \odot \hat{\mathbf{y}}_i
+
\mathbf{b}_{u(i)},
$$

where $u(i)$ is the subject identity for sample $i$, $\mathbf{s}_{u(i)}$ is the subject-specific scale vector, and $\mathbf{b}_{u(i)}$ is the subject-specific bias vector.

To reduce overfitting when the support set is small, the subject-specific parameters are shrunk toward global estimates:

$$
\mathbf{s}_u =
\frac{n_u}{n_u+\tau}\hat{\mathbf{s}}_u
+
\frac{\tau}{n_u+\tau}\mathbf{s}_0,
\quad
\mathbf{b}_u =
\frac{n_u}{n_u+\tau}\hat{\mathbf{b}}_u
+
\frac{\tau}{n_u+\tau}\mathbf{b}_0.
$$

Here, $n_u$ is the number of support samples for subject $u$, $\tau$ is the shrinkage strength, and $(\mathbf{s}_0,\mathbf{b}_0)$ are global calibration priors.

The overall training objective combines regression, uncertainty, auxiliary decision, router, credibility, and tail-safety terms:

$$
\mathcal{L}
=
\mathcal{L}_{reg}
+
\lambda_{NLL}\mathcal{L}_{NLL}
+
\lambda_{aux}\mathcal{L}_{aux}
+
\lambda_{router}\mathcal{L}_{router}
+
\lambda_{cred}\mathcal{L}_{cred}
+
\lambda_{tail}\mathcal{L}_{tail}.
$$

The heteroscedastic negative log-likelihood term is:

$$
\mathcal{L}_{NLL}
=
\frac{1}{2}
\sum_i
\sum_{d \in \{sbp, dbp\}}
\left[
\exp(-s_{i,d})(y_{i,d}-\mu_{i,d})^2
+
s_{i,d}
\right].
$$

The upper-tail asymmetric loss emphasizes clinically undesirable underestimation:

$$
\mathcal{L}_{tail}
=
\frac{1}{N}
\sum_{i=1}^{N}
\omega^{tail}_i
\sum_{d \in \{sbp, dbp\}}
\left[
\eta_d \mathrm{ReLU}(y_{i,d}-\hat{y}_{i,d})
+
\mathrm{ReLU}(\hat{y}_{i,d}-y_{i,d})
\right],
$$

where $\eta_d > 1$ increases the penalty for underestimating high BP.

Split conformal prediction is then used to report interval reliability. For target dimension $d \in \{sbp, dbp\}$, the calibration residual is:

$$
s^{(d)}_j =
\left|
y^{(d)}_j - \tilde{y}^{(d)}_j
\right|.
$$

Given target miscoverage $\alpha$, the conformal interval for sample $i$ is:

$$
\mathcal{I}^{(d)}_i =
\left[
\tilde{y}^{(d)}_i - q^{(d)}_{1-\alpha},
\quad
\tilde{y}^{(d)}_i + q^{(d)}_{1-\alpha}
\right],
$$

where $q^{(d)}_{1-\alpha}$ is the empirical $(1-\alpha)$ quantile of calibration residuals.

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
