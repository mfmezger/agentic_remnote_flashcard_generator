from pydantic_ai import BinaryContent

from flashcard_generator.prompts import SYSTEM_PROMPT
from flashcard_generator.utils.gemini_agent import initialize_agent
from pathlib import Path


def main() -> None:
    # generate results folder
    agent = initialize_agent(prompt=SYSTEM_PROMPT)
    Path("results").mkdir(exist_ok=True)

    # iterate over the data folder
    files = sorted(Path("data").glob(pattern="*.pdf"))
    results: list[str] = ["# Flashcards\n"]

    for file in files:
        print(f"Processing file: {file.name}")

        with open(file=file, mode="rb") as f:
            file_content = f.read()

        result = agent.run_sync(
            [
                "Please generate flashcards for the following document:",
                BinaryContent(data=file_content, media_type="application/pdf"),
            ]
        )  # pyright: ignore[reportArgumentType]

        # adding the name of the file as a header
        results.append(f"## Flashcards for {file.name}\n")

        for r in result.output.flashcards:
            results.append("- " + r.question + " == " + r.answer)

    # save the outputs as one markdown file.
    with open(file="results/flashcards.md", mode="w") as f:
        f.write("\n".join(results))


if __name__ == "__main__":
    main()
