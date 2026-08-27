"""Score movie reviews from the command line.

Examples:
    python cli.py "An absolute masterpiece, I loved every minute."
    python cli.py --file reviews.txt
    python cli.py --csv reviews.csv --column review --out scored.csv
    python cli.py --interactive
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

from src.predict import SentimentAnalyzer

BAR_WIDTH = 24


def render(result: dict) -> str:
    filled = round(result["positive_score"] * BAR_WIDTH)
    bar = "#" * filled + "." * (BAR_WIDTH - filled)
    return (
        f"{result['label'].upper():<9} conf {result['confidence']:.1%}  "
        f"[{bar}] pos={result['positive_score']:.3f}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Movie review sentiment CLI")
    parser.add_argument("text", nargs="*", help="review text")
    parser.add_argument("--file", help="text file, one review per line")
    parser.add_argument("--csv", help="CSV file of reviews")
    parser.add_argument("--column", default="review", help="CSV column name")
    parser.add_argument("--out", help="write CSV results here")
    parser.add_argument("--interactive", action="store_true")
    args = parser.parse_args()

    analyzer = SentimentAnalyzer()

    if args.interactive:
        print("Type a review (blank line or Ctrl-C to quit).")
        try:
            while True:
                line = input("> ").strip()
                if not line:
                    break
                print(render(analyzer.predict(line)))
        except (KeyboardInterrupt, EOFError):
            print()
        return

    texts: list[str] = []
    if args.text:
        texts.append(" ".join(args.text))
    if args.file:
        texts += [l.strip() for l in Path(args.file).read_text(encoding="utf-8").splitlines() if l.strip()]
    if args.csv:
        with open(args.csv, newline="", encoding="utf-8") as fh:
            texts += [row[args.column] for row in csv.DictReader(fh)]

    if not texts:
        parser.print_help()
        sys.exit(1)

    results = analyzer.predict_batch(texts)
    for text, result in zip(texts, results):
        preview = text if len(text) <= 70 else text[:67] + "..."
        print(f"{render(result)}  | {preview}")

    if args.out:
        with open(args.out, "w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=["text", "label", "confidence", "positive_score"])
            writer.writeheader()
            for r in results:
                writer.writerow({k: r[k] for k in writer.fieldnames})
        print(f"\nSaved {len(results)} rows to {args.out}")


if __name__ == "__main__":
    main()
