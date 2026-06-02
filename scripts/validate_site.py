import argparse
from pathlib import Path
import re
import sys


REQUIRED_HEADINGS = [
    "## Introduction",
    "## Data Cleaning and Exploratory Data Analysis",
    "## Assessment of Missingness",
    "## Hypothesis Testing",
    "## Framing a Prediction Problem",
    "## Baseline Model",
    "## Final Model",
    "## Fairness Analysis",
]

FORBIDDEN_PATTERNS = [
    "Content coming after",
    "TODO",
    "TBD",
    "Your Title Here",
    "Add name(s) here",
]

FINAL_INCOMPLETE_PATTERNS = [
    "This section will be completed for the final project.",
    "will be completed for the final project",
    "Current status: this analysis is still pending.",
    "Current status: the final model has not been trained yet.",
    "Current status: this analysis is still pending because it should use the final fitted model.",
    "Planned but not yet run.",
    "Planned but not yet trained.",
    "Planned but depends on the final model.",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate the public GitHub Pages report.")
    parser.add_argument(
        "--final",
        action="store_true",
        help="Require final-project sections to be complete.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[1]
    readme_path = repo_root / "README.md"

    if not readme_path.exists():
        print("README.md is missing.")
        return 1

    readme = readme_path.read_text(encoding="utf-8")
    failures = []

    for heading in REQUIRED_HEADINGS:
        if heading not in readme:
            failures.append(f"Missing required heading: {heading}")

    for pattern in FORBIDDEN_PATTERNS:
        if pattern in readme:
            failures.append(f"Found unfinished placeholder text: {pattern}")

    iframe_sources = re.findall(r'<iframe[^>]+src="([^"]+)"', readme)
    for source in iframe_sources:
        asset_path = repo_root / source
        if not asset_path.exists():
            failures.append(f"Missing iframe asset: {source}")

    if "label:*" not in readme:
        failures.append("README should explicitly state that label:* columns are excluded.")

    if args.final:
        for pattern in FINAL_INCOMPLETE_PATTERNS:
            if pattern in readme:
                failures.append(f"Final-project content still incomplete: {pattern}")

    if failures:
        print("Site validation failed:")
        for failure in failures:
            print(f"- {failure}")
        return 1

    print("Site validation passed.")
    print(f"Checked {len(REQUIRED_HEADINGS)} required headings and {len(iframe_sources)} iframe assets.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
