import os
from pathlib import Path

ROOT = Path(
    os.environ.get(
        "AQM_MIMIC_BP_ROOT",
        os.environ.get("AQM_MIMIC_BP_DATA_ROOT", str(Path(__file__).resolve().parent / "data" / "raw" / "MIMIC-BP")),
    )
)
OUT_TXT = ROOT / "mimic_bp_inventory.txt"

MAX_SHOW_PER_DIR = 30  # 每个目录最多展示多少个文件名


def list_files(folder: Path):
    if not folder.exists():
        return []
    return sorted([p for p in folder.iterdir() if p.is_file()], key=lambda x: x.name.lower())


def write_section(f, title: str):
    f.write("=" * 80 + "\n")
    f.write(title + "\n")
    f.write("=" * 80 + "\n")


def main():
    subdirs = [
        "downloads",
        "ppg",
        "ecg",
        "abp",
        "labels",
        "resp",
        "splits",
        "metadata",
    ]

    with open(OUT_TXT, "w", encoding="utf-8") as f:
        write_section(f, f"MIMIC-BP inventory under: {ROOT}")

        f.write(f"ROOT exists: {ROOT.exists()}\n\n")

        # 列出一级目录结构
        write_section(f, "Top-level entries")
        if ROOT.exists():
            for p in sorted(ROOT.iterdir(), key=lambda x: (not x.is_dir(), x.name.lower())):
                kind = "DIR " if p.is_dir() else "FILE"
                f.write(f"[{kind}] {p.name}\n")
        else:
            f.write("ROOT does not exist.\n")

        f.write("\n")

        # 分目录列文件
        for dname in subdirs:
            folder = ROOT / dname
            write_section(f, f"Directory: {folder}")
            if not folder.exists():
                f.write("Directory does not exist.\n\n")
                continue

            files = list_files(folder)
            f.write(f"Exists: True\n")
            f.write(f"File count: {len(files)}\n\n")

            if len(files) == 0:
                f.write("No files found.\n\n")
                continue

            f.write(f"Showing up to first {MAX_SHOW_PER_DIR} files:\n")
            for p in files[:MAX_SHOW_PER_DIR]:
                f.write(f"{p.name}\n")

            if len(files) > MAX_SHOW_PER_DIR:
                f.write(f"... ({len(files) - MAX_SHOW_PER_DIR} more files not shown)\n")
            f.write("\n")

        # 专门检查 splits
        write_section(f, "Split file check")
        split_dir = ROOT / "splits"
        expected = ["train_subjects.txt", "val_subjects.txt", "test_subjects.txt"]
        if split_dir.exists():
            for name in expected:
                path = split_dir / name
                f.write(f"{name}: {'FOUND' if path.exists() else 'MISSING'}\n")
        else:
            f.write("splits directory does not exist.\n")

        f.write("\n")

        # 全局递归查找 subject txt
        write_section(f, "Recursive search for subject split files")
        for name in expected:
            matches = sorted(ROOT.rglob(name))
            if matches:
                f.write(f"{name}: FOUND {len(matches)} match(es)\n")
                for m in matches:
                    f.write(f"  - {m}\n")
            else:
                f.write(f"{name}: NOT FOUND\n")
        f.write("\n")

        # 递归抽样查找 ppg/ecg/labels 文件名模式
        write_section(f, "Recursive filename pattern examples")
        patterns = ["*_ppg.npy", "*_ecg.npy", "*_abp.npy", "*_labels.npy"]
        for pat in patterns:
            matches = sorted(ROOT.rglob(pat))
            f.write(f"Pattern {pat}: {len(matches)} file(s)\n")
            for m in matches[:10]:
                f.write(f"  - {m.name}\n")
            if len(matches) > 10:
                f.write(f"  ... ({len(matches) - 10} more)\n")
            f.write("\n")

    print(f"Done. Inventory saved to:\n{OUT_TXT}")


if __name__ == "__main__":
    main()
