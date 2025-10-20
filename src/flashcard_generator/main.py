from pydantic_ai import BinaryContent

from flashcard_generator.prompts import SYSTEM_PROMPT
from flashcard_generator.utils.gemini_agent import initialize_agent
from pathlib import Path


def main():
    # generate results folder
    agent = initialize_agent(prompt=SYSTEM_PROMPT)
    Path("results").mkdir(exist_ok=True)

    # iterate over the data folder
    files = sorted(Path("data").glob(pattern="*.pdf"))
    results: list[str] = []

    for file in files:
        print(f"Processing file: {file.name}")

        with open(file, "rb") as f:
            file_content = f.read()

        result = agent.run_sync(
            [
                "Please generate flashcards for the following document:",
                BinaryContent(data=file_content, media_type="application/pdf"),
            ]
        )  # pyright: ignore[reportArgumentType]

        results.append(result.output)

    # save the outputs as one markdown file.
    with open("results/flashcards.md", "w") as f:
        for file, result in zip(files, results):
            f.write(result)
            f.write("\n\n")


if __name__ == "__main__":
    main()
