# RemNote flashcard generator

Turns PDFs (lecture slides, papers) into flashcards in RemNote's Markdown format
(`- question == answer`) using Gemini via pydantic-ai.

## Usage

```bash
echo 'GEMINI_API_KEY=...' > .env   # or export it
mkdir -p data && cp ~/slides/*.pdf data/
uv run flashcard-generator          # writes results/flashcards.md
uv run flashcard-generator --data other/ --output results/other.md
```

Paste the resulting Markdown into RemNote; each file gets its own `##` heading.

`data/`, `results/` and `.env` are gitignored. Finished decks worth keeping live in
`flashcards/` (e.g. `flashcards/metrics.md`).
