# Recommended GitHub Settings

Repository name:

```text
SAQM-MedFuse
```

Short description:

```text
Official reproducibility code and aggregate results for SAQM-MedFuse, a safety-aware quality-driven multimodal framework for cuffless blood pressure estimation and risk stratification from PPG and ECG.
```

Topics:

```text
cuffless-blood-pressure
blood-pressure-estimation
physiological-signals
ppg
ecg
multimodal-learning
medical-ai
pytorch
uncertainty-estimation
conformal-prediction
mimic-bp
reproducibility
```

Website:

```text
https://github.com/yazyeah/SAQM-MedFuse
```

License:

```text
MIT License
```

The repository already contains a `LICENSE` file, so GitHub should recognize
the license automatically after the push.

Release recommendation:

- Use `v0.1.0` for the current initial reproducibility package if the repository
  is already public but major-module ablation results are still running.
- Use `v1.0.0` only after the paper-facing main, comparison, and major-module
  ablation summaries are complete and checked.

Packages:

Do not publish a GitHub Package for the current release. This repository is a
research reproducibility package, not a reusable pip package or Docker image.
Add packages later only if you create a stable Python package or containerized
reproduction workflow.
