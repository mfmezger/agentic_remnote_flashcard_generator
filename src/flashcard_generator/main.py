"""Generate RemNote flashcards for every PDF in a folder."""

import argparse
from pathlib import Path

from dotenv import load_dotenv
from pydantic_ai import BinaryContent

from flashcard_generator.agent import initialize_agent


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data", type=Path, default=Path("data"), help="folder with PDFs"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/flashcards.md"),
        help="markdown output file",
    )
    args = parser.parse_args()

    files = sorted(args.data.glob("*.pdf"))
    if not files:
        raise SystemExit(f"No PDFs found in {args.data}/")

    load_dotenv(override=True)
    agent = initialize_agent()
    lines = ["# Flashcards", ""]

    for file in files:
        print(f"Processing file: {file.name}")
        result = agent.run_sync(
            [
                "Please generate flashcards for the following document:",
                BinaryContent(data=file.read_bytes(), media_type="application/pdf"),
            ]
        )
        lines += [f"## Flashcards for {file.name}", ""]
        lines += [
            f"- {card.question} == {card.answer}" for card in result.output.flashcards
        ]
        lines.append("")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(lines))
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
