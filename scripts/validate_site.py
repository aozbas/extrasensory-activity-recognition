import argparse
from pathlib import Path
import re
import sys


SITE_HEADINGS = [
    "## Introduction",
    "## Data Cleaning and Exploratory Data Analysis",
    "## Assessment of Missingness",
    "## Hypothesis Testing",
    "## Framing a Prediction Problem",
    "## Baseline Model",
    "## Final Model",
    "## Fairness Analysis",
]

PLACEHOLDERS = [
    "Content coming after",
    "TODO",
    "TBD",
    "Your Title Here",
    "Add name(s) here",
]

FINAL_PLACEHOLDERS = [
    "This section will be completed for the final project.",
    "will be completed for the final project",
    "Current status: this analysis is still pending.",
    "Current status: the final model has not been trained yet.",
    "Current status: this analysis is still pending because it should use the final fitted model.",
    "Planned but not yet run.",
    "Planned but not yet trained.",
    "Planned but depends on the final model.",
]


def get_args():
    parser = argparse.ArgumentParser(description="Check the project website.")
    parser.add_argument("--final", action="store_true")
    return parser.parse_args()


def main():
    args = get_args()
    repo = Path(__file__).resolve().parents[1]
    readme_path = repo / "README.md"
    errors = []

    if not readme_path.exists():
        print("README.md is missing.")
        return 1

    readme = readme_path.read_text(encoding="utf-8")

    for heading in SITE_HEADINGS:
        if heading not in readme:
            errors.append(f"Missing required heading: {heading}")

    for text in PLACEHOLDERS:
        if text in readme:
            errors.append(f"Found unfinished placeholder text: {text}")

    if args.final:
        for text in FINAL_PLACEHOLDERS:
            if text in readme:
                errors.append(f"Final-project content still incomplete: {text}")

    if "label:*" not in readme:
        errors.append("README should explicitly state that label:* columns are excluded.")

    iframe_sources = re.findall(r'<iframe[^>]+src="([^"]+)"', readme)
    for source in iframe_sources:
        if not (repo / source).exists():
            errors.append(f"Missing iframe asset: {source}")

    if errors:
        print("Site validation failed:")
        for error in errors:
            print(f"- {error}")
        return 1

    print("Site validation passed.")
    print(f"Checked {len(SITE_HEADINGS)} required headings and {len(iframe_sources)} iframe assets.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
