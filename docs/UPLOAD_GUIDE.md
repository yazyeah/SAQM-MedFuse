# GitHub Upload Guide

This guide assumes that you upload the clean release directory, not the full
local experiment workspace.

## 1. Confirm the release directory

Open PowerShell:

```powershell
cd <PATH_TO_THIS_RELEASE_DIRECTORY>
```

Inspect the files that will be published:

```powershell
Get-ChildItem -Recurse | Select-Object FullName
```

Run the built-in checks:

```powershell
.\scripts\check_release_safety.ps1
.\scripts\check_experiment_completion.ps1
```

If the completion check reports incomplete experiment rows, do not upload those
summary files as final paper results. Finish the corresponding runs, refresh the
summary tables, and copy only the completed aggregate CSV/JSON files into
`results/`.

The repository should contain source code, wrapper scripts, aggregate results,
documentation, and selected paper figures. It should not contain local data,
model checkpoints, Python virtual environments, downloaded PDFs, or generated
per-subject outputs.

## 2. Confirm ignored files

```powershell
git init
git status --ignored
```

Check that the following are not staged:

- `data/`, `Datasets/`, `outputs/`
- `*.pt`, `*.pth`, `*.ckpt`, `*.npy`, `*.npz`, `*.zip`
- per-subject split and prediction files
- local path files such as `configs/paths.local.ps1`

## 3. Create the first commit

```powershell
git add .
git status
git commit -m "Release SAQM-MedFuse reproducibility package"
```

## 4. Create a GitHub repository

Create an empty GitHub repository, for example:

```text
https://github.com/<YOUR_GITHUB_USERNAME>/SAQM-MedFuse
```

Do not initialize the GitHub repository with another README if you already have
the local README in this release directory.

## 5. Connect and push

```powershell
git branch -M main
git remote add origin https://github.com/<YOUR_GITHUB_USERNAME>/SAQM-MedFuse.git
git push -u origin main
```

If Git asks for authentication, use GitHub web login, GitHub CLI, or a personal
access token according to your local Git configuration.

## 6. After upload

Open the GitHub page and verify:

- README renders correctly.
- `results/paper_summary/` contains only aggregate CSV files.
- `results/figures/` contains selected figures only.
- No raw MIMIC-BP data or model checkpoints are visible.
- The repository URL in `CITATION.cff` is updated from the placeholder.

## 7. Optional release tag

```powershell
git tag v1.0.0
git push origin v1.0.0
```

Use a release tag when the paper tables are finalized.
