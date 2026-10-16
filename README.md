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
`flashcards/` (excluded from the `typos` hook, which would "fix" card content).

## Decks

| Deck | Topic |
| --- | --- |
| [metrics](flashcards/metrics.md) | Classic ML evaluation metrics |
| [llm-evals](flashcards/llm-evals.md) | Evaluating LLM systems, judges, benchmarks |
| [llm-security](flashcards/llm-security.md) | Prompt injection, jailbreaks, OWASP LLM Top 10, guardrails |
| [agents-tool-use](flashcards/agents-tool-use.md) | Agent patterns, tool calling, MCP, context engineering |
| [rag](flashcards/rag.md) | Chunking, retrieval, reranking, ANN indexes, RAG evals |
| [inference-serving](flashcards/inference-serving.md) | KV cache, batching, speculative decoding, quantization |
| [transformer-internals](flashcards/transformer-internals.md) | Attention, positional encodings, tokenization, MoE |
| [fine-tuning-alignment](flashcards/fine-tuning-alignment.md) | SFT, LoRA/QLoRA, RLHF, DPO, GRPO |
| [llmops-observability](flashcards/llmops-observability.md) | OpenTelemetry, metrics, SLOs, caching, MLflow |
| [eu-ai-act](flashcards/eu-ai-act.md) | EU AI Act roles, risk classes, obligations, timeline |
