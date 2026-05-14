import os
from pathlib import Path
import re
import random
import json

ROOT = Path(
    os.environ.get(
        "AQM_MIMIC_BP_ROOT",
        os.environ.get("AQM_MIMIC_BP_DATA_ROOT", str(Path(__file__).resolve().parent / "data" / "raw" / "MIMIC-BP")),
    )
)
SPLIT_DIR = ROOT / "splits"

SEED = 42
TRAIN_RATIO = 0.70
VAL_RATIO = 0.10
CALIB_RATIO = 0.10
TEST_RATIO = 0.10

REQUIRED_MODALITIES = ["ppg", "ecg", "labels"]  # abp 可保留为参考，但不作为主输入要求


def parse_subjects(folder: Path, suffix: str):
    pattern = re.compile(rf"^(p\d{{6}})_{suffix}\.npy$", re.IGNORECASE)
    subjects = set()
    if not folder.exists():
        return subjects
    for p in folder.iterdir():
        if p.is_file():
            m = pattern.match(p.name)
            if m:
                subjects.add(m.group(1))
    return subjects


def write_list(path: Path, items):
    with open(path, "w", encoding="utf-8") as f:
        for x in items:
            f.write(f"{x}\n")


def main():
    assert abs(TRAIN_RATIO + VAL_RATIO + CALIB_RATIO + TEST_RATIO - 1.0) < 1e-8

    SPLIT_DIR.mkdir(parents=True, exist_ok=True)

    subject_sets = {}
    for mod in REQUIRED_MODALITIES:
        folder = ROOT / mod
        subject_sets[mod] = parse_subjects(folder, mod)
        print(f"{mod}: {len(subject_sets[mod])} subjects found")

    common_subjects = sorted(set.intersection(*subject_sets.values()))
    print(f"Common subjects across {REQUIRED_MODALITIES}: {len(common_subjects)}")

    if len(common_subjects) == 0:
        raise RuntimeError("No common subjects found across required modalities.")

    rng = random.Random(SEED)
    rng.shuffle(common_subjects)

    n = len(common_subjects)
    n_train = int(round(n * TRAIN_RATIO))
    n_val = int(round(n * VAL_RATIO))
    n_calib = int(round(n * CALIB_RATIO))
    n_test = n - n_train - n_val - n_calib

    train_subjects = common_subjects[:n_train]
    val_subjects = common_subjects[n_train:n_train + n_val]
    calib_subjects = common_subjects[n_train + n_val:n_train + n_val + n_calib]
    test_subjects = common_subjects[n_train + n_val + n_calib:]

    write_list(SPLIT_DIR / "train_subjects.txt", train_subjects)
    write_list(SPLIT_DIR / "val_subjects.txt", val_subjects)
    write_list(SPLIT_DIR / "calib_subjects.txt", calib_subjects)
    write_list(SPLIT_DIR / "test_subjects.txt", test_subjects)

    summary = {
        "root": str(ROOT),
        "seed": SEED,
        "required_modalities": REQUIRED_MODALITIES,
        "num_common_subjects": len(common_subjects),
        "num_train": len(train_subjects),
        "num_val": len(val_subjects),
        "num_calib": len(calib_subjects),
        "num_test": len(test_subjects),
        "example_train_subjects": train_subjects[:10],
        "example_val_subjects": val_subjects[:10],
        "example_calib_subjects": calib_subjects[:10],
        "example_test_subjects": test_subjects[:10],
    }

    with open(SPLIT_DIR / "split_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("\nSplit files generated in:")
    print(SPLIT_DIR)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
