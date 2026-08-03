from __future__ import annotations

import argparse
from pathlib import Path
import sys

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ifdr_yolo.eval.research_visualization import generate_research_figures


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate publication-ready IFDR research figures.")
    parser.add_argument("--runs", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    generated = generate_research_figures(args.runs, args.output)
    for path in generated:
        print(path)


if __name__ == "__main__":
    main()
