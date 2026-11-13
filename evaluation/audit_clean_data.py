
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

DATASETS = {
    "Training": ROOT / "data/cleaned/training_clean.txt",
    "Validation": ROOT / "data/cleaned/validation_clean.txt",
}

ARTIFACTS = ["@-@", "@.@", "@,@", "= ="]


def audit(name, path):
    if not path.exists():
        raise FileNotFoundError(path)

    text = path.read_text(encoding="utf-8")

    lines = text.splitlines()
    nonempty = [line for line in lines if line.strip()]

    print(f"\n===== {name} DATASET =====")
    print("Characters:", f"{len(text):,}")
    print("Lines:", f"{len(lines):,}")
    print("Non-empty lines:", f"{len(nonempty):,}")
    print("Words:", f"{len(text.split()):,}")
    print("Unique non-empty lines:", f"{len(set(nonempty)):,}")
    print("Duplicate non-empty lines:", f"{len(nonempty) - len(set(nonempty)):,}")
    print("Replacement characters:", text.count("\ufffd"))

    for artifact in ARTIFACTS:
        print(f"{artifact} occurrences:", text.count(artifact))

    problems = []

    if "\ufffd" in text:
        problems.append("Replacement characters remain.")

    for artifact in ARTIFACTS:
        if artifact in text:
            problems.append(f"Artifact remains: {artifact}")

    if problems:
        raise ValueError("\n".join(problems))

    print("Result: PASSED")


def main():
    for name, path in DATASETS.items():
        audit(name, path)

    print("\nBoth cleaned datasets passed the basic artifact audit.")


if __name__ == "__main__":
    main()
