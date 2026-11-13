
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

FILES = {
    ROOT / "data/input.txt": ROOT / "data/cleaned/training_clean.txt",
    ROOT / "data/wikitext_validation.txt": ROOT / "data/cleaned/validation_clean.txt",
}


def clean_text(text):
    # Replace corrupted Unicode characters
    text = text.replace("\ufffd", " ")

    # Fix WikiText punctuation artifacts
    text = text.replace("@-@", "-")
    text = text.replace("@.@", ". ")
    text = text.replace("@,@", ", ")

    cleaned_lines = []

    for line in text.splitlines():
        # Remove heading markers at line boundaries
        line = re.sub(r"^\s*(?:=\s*)+", "", line)
        line = re.sub(r"(?:\s*=\s*)+$", "", line)

        # Remove trailing whitespace
        cleaned_lines.append(line.rstrip())

    return "\n".join(cleaned_lines) + "\n"


def main():
    for source, destination in FILES.items():
        if not source.exists():
            raise FileNotFoundError(source)

        print(f"\nReading: {source.name}")

        raw_text = source.read_text(encoding="utf-8", errors="replace")
        cleaned_text = clean_text(raw_text)

        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(cleaned_text, encoding="utf-8")

        print(f"Original characters: {len(raw_text):,}")
        print(f"Cleaned characters:  {len(cleaned_text):,}")
        print(f"Saved: {destination.relative_to(ROOT)}")

    print("\nBoth datasets cleaned successfully.")


if __name__ == "__main__":
    main()
