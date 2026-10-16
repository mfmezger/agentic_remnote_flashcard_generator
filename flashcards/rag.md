# Flashcards: Retrieval-Augmented Generation (RAG)

## Why RAG vs Fine-Tuning vs Long Context

- What core problem does RAG solve for LLMs? == It grounds generation in retrieved external documents, giving the model access to private or fresh knowledge and reducing hallucination without retraining.
- When is fine-tuning preferable to RAG? == When you need to change model behavior, style, output format, or task skill; fine-tuning is unreliable for injecting frequently changing factual knowledge.
- Why prefer RAG over fine-tuning for knowledge that changes often? == Updating knowledge only requires re-indexing documents rather than retraining, and answers can cite their sources.
- What are the main downsides of putting an entire corpus into a long context window instead of retrieving? == Higher cost and latency per query, a hard cap on corpus size, and weaker use of relevant facts buried in very long inputs.
- When can long context reasonably replace RAG? == When the relevant material is small enough to fit in the window and the per-query cost and latency of processing it all are acceptable.
- How does RAG support access control better than fine-tuning? == Retrieval can filter documents by the user's permissions at query time, which is impossible once knowledge is baked into model weights.

## Pipeline Stages

- What are the offline (indexing) stages of a RAG pipeline? == Ingestion, parsing, chunking, embedding, and indexing documents into a searchable store.
- What are the online (query-time) stages of a RAG pipeline? == Query processing, retrieval, optional reranking, prompt assembly with the retrieved context, and generation.
- What does the parsing stage do and why is it critical? == It converts raw files (PDF, HTML, slides) into clean text plus structure like headings, tables, and reading order; parsing errors propagate to every later stage.
- Why use a two-stage retrieve-then-rerank design? == Cheap first-stage retrieval achieves high recall over the whole corpus, then a slower, more accurate reranker improves precision on a small candidate set.
- What metadata should be stored alongside each chunk? == Source document ID, location (page, section), timestamps, and access attributes, enabling filtering, citations, and incremental updates.

## Chunking Strategies

- Why are documents chunked before embedding? == Embedding models have input limits, one vector represents long text poorly, and smaller units let retrieval return only the relevant passages.
- What is the chunk size trade-off? == Small chunks match precisely but lose surrounding context; large chunks keep context but dilute the embedding and waste prompt tokens.
- What is fixed-size chunking with overlap? == Splitting text into chunks of N tokens where consecutive chunks share some tokens, so content near a boundary appears intact in at least one chunk.
- What is recursive chunking? == Splitting by a prioritized list of separators (e.g., paragraphs, then lines, then spaces) until each piece fits the size limit, preserving natural boundaries where possible.
- What is semantic chunking? == Placing chunk boundaries where embedding similarity between consecutive sentences drops, so each chunk stays topically coherent.
- What is document-structure-aware chunking? == Splitting along the document's own structure, such as headings, sections, Markdown elements, code functions, or table boundaries.
- What is late chunking? == Embedding the whole document with a long-context embedding model first, then pooling token embeddings within each chunk span, so every chunk vector reflects document-wide context.
- What is contextual retrieval (chunk context prepending)? == An LLM writes a short chunk-specific context explaining where the chunk fits in its document, which is prepended to the chunk before embedding and BM25 indexing.
- Why does prepending context such as the document title or section heading help retrieval? == It restores information lost by splitting, so a chunk like "revenue grew 3%" becomes findable for queries naming the company and period.
- What is parent-document (small-to-big) retrieval? == Matching on small, precise chunks but passing their larger parent section to the LLM, combining retrieval precision with generation context.

## Embeddings

- What is a bi-encoder? == A model that encodes queries and documents independently into vectors, so document embeddings can be precomputed and compared with a cheap similarity function.
- How is cosine similarity related to dot product? == Cosine similarity is the dot product of L2-normalized vectors; for unit-length vectors the two are identical and produce the same ranking.
- When do cosine and dot-product rankings differ? == When vectors are not normalized: dot product also rewards vector magnitude, while cosine considers only direction.
- What are the benefits of normalizing embeddings? == Dot product then equals cosine, fast inner-product search can be used, and Euclidean distance ranking also matches cosine ranking.
- What are Matryoshka embeddings? == Embeddings trained so that prefixes of the vector (e.g., the first 256 of 1024 dimensions) are themselves useful embeddings, allowing truncation to trade accuracy for storage and speed.
- What must you do after truncating a Matryoshka embedding? == Re-normalize the truncated vector before comparing with cosine or dot product.
- What are asymmetric query/document prefixes? == Task instructions some models require, like "query: " vs "passage: " (E5), because short queries and long passages are embedded differently; omitting them silently degrades retrieval.
- What happens if you switch embedding models for queries but not the indexed documents? == Retrieval breaks, because vectors from different models live in incompatible spaces; changing models requires re-embedding the whole corpus.

## Sparse Retrieval (BM25)

- What is BM25? == A sparse lexical ranking function that scores documents by query-term matches, weighting term frequency with saturation, inverse document frequency, and document length normalization.
- What is the BM25 per-term score? == IDF(t) * tf * (k1 + 1) / (tf + k1 * (1 - b + b * docLen / avgDocLen)), summed over all query terms.
- What does the k1 parameter control in BM25? == Term-frequency saturation: how quickly extra occurrences stop adding score; k1 = 0 ignores tf, larger k1 approaches linear tf, typically 1.2 to 2.0.
- What does the b parameter control in BM25? == Document length normalization from 0 (none) to 1 (full); higher b penalizes long documents more for the same term count, typically 0.75.
- What does IDF contribute in BM25? == It upweights rare terms and downweights common ones, since matching a rare term is stronger evidence of relevance.
- Where does BM25 typically beat dense retrieval? == Exact-match queries with rare keywords, product codes, IDs, error messages, names, and domain jargon the embedding model never learned.
- What is learned sparse retrieval such as SPLADE? == A model that outputs weighted vocabulary terms, including expansion terms absent from the text, which can be served from a standard inverted index.

## Hybrid Search and Reciprocal Rank Fusion

- What is hybrid search? == Combining sparse (BM25) and dense (embedding) retrieval so both exact lexical matches and semantic matches contribute to the final ranking.
- What is the Reciprocal Rank Fusion (RRF) formula? == RRF(d) = sum over result lists of 1 / (k + rank_i(d)), with ranks starting at 1 and k typically 60.
- Why does RRF use ranks instead of raw scores? == BM25 scores and cosine similarities have incomparable scales and distributions; ranks need no normalization or calibration.
- What is the role of the constant k in RRF? == It damps the advantage of top positions; larger k flattens differences between ranks, so documents found by several retrievers win out.
- What is the main alternative to RRF for hybrid fusion? == Weighted score fusion: normalize each retriever's scores (e.g., min-max) and combine them with a tunable weight, which is more tunable but sensitive to score distributions.

## Reranking and Late Interaction

- What is a cross-encoder? == A model that takes query and document together as one input and outputs a single relevance score for the pair.
- Why are cross-encoders more accurate than bi-encoders? == They jointly attend over query and document tokens instead of compressing each into a single vector independently.
- Why can't cross-encoders be used for first-stage retrieval over large corpora? == Nothing can be precomputed: every query needs a full forward pass per candidate document, which is far too slow for millions of documents.
- What is a typical reranking configuration? == Retrieve tens to a few hundred candidates with hybrid search, rerank them with a cross-encoder, and pass the top handful to the LLM.
- What is ColBERT's late interaction? == Query and document are encoded independently into per-token embeddings, and relevance is the sum over query tokens of the maximum similarity to any document token (MaxSim).
- What is the main trade-off of ColBERT versus single-vector retrieval? == Higher accuracy with precomputable document embeddings, but a much larger index because it stores one vector per token.

## ANN Indexes

- Why use approximate nearest neighbor (ANN) indexes? == Exact k-NN search scales linearly with corpus size; ANN trades a small loss in recall for orders-of-magnitude lower latency.
- What is HNSW? == A multi-layer proximity graph where sparse upper layers allow long jumps and the bottom layer contains all vectors; search greedily descends layers toward the query.
- What does the HNSW parameter M control? == The maximum number of neighbor links per node per layer (typically 2M on the bottom layer); higher M improves recall but increases memory use and build time.
- What does the HNSW parameter efConstruction control? == The candidate list size during index building; higher values produce a better-connected graph and higher recall at the cost of slower indexing.
- What does the HNSW parameter efSearch control? == The candidate list size at query time; higher values raise recall and latency, and it can be tuned per query without rebuilding the index.
- What is an IVF index? == It clusters vectors with k-means into nlist cells and, at query time, searches only the nprobe cells whose centroids are nearest the query.
- What is the nprobe trade-off in IVF? == Probing more cells raises recall but increases latency; too few cells misses neighbors that sit just across cluster boundaries.
- What is product quantization (PQ)? == Splitting each vector into m subvectors and replacing each with the ID of its nearest centroid in a per-subspace codebook, compressing vectors to a few bytes with fast approximate distances.
- What is the main trade-off of product quantization? == Large memory savings and speed but lossy distances that reduce recall, commonly mitigated by rescoring top candidates with full-precision vectors.
- How should ANN index quality be measured? == Recall@k relative to exact brute-force neighbors, plotted against latency or QPS to choose an operating point.

## Metadata Filtering

- What is pre-filtering in vector search? == Applying metadata constraints first and searching only among matching vectors, so all k results satisfy the filter.
- What is post-filtering in vector search? == Running vector search first and discarding non-matching results afterward; simple, but selective filters can leave fewer than k results or none.
- Why is pre-filtering hard with HNSW? == Restrictive filters can break graph traversal connectivity or degrade toward brute force, so engines use filter-aware traversal or switch to exact search for small subsets.
- How is post-filtering's result shortfall usually mitigated? == Over-fetching many more than k candidates before filtering, though highly selective filters may still return too few.

## Query Transformation

- What is query rewriting? == Using an LLM to reformulate the user query before retrieval, e.g., turning a follow-up with pronouns into a standalone question using chat history.
- What is multi-query retrieval? == Generating several paraphrases of the query, retrieving for each, and fusing the results (e.g., with RRF) to improve recall.
- What is HyDE? == Hypothetical Document Embeddings: an LLM writes a hypothetical answer and its embedding is used for retrieval, since answer-like text lies closer to relevant passages than a short question.
- What are the risks of HyDE? == The hypothetical answer may contain hallucinated specifics that steer retrieval toward wrong content, and it adds an LLM call of latency.
- What is query decomposition? == Breaking a complex or multi-hop question into sub-questions, retrieving for each, and combining the evidence for the final answer.
- What is query routing? == Choosing which index, data source, or tool to query based on the question, e.g., a SQL database versus a document store.

## GraphRAG and Agentic RAG

- What is GraphRAG? == RAG over a knowledge graph of entities and relationships, often extracted from documents by an LLM, so retrieval can follow connections across documents.
- How does Microsoft GraphRAG answer global questions? == It clusters the entity graph into hierarchical communities (Leiden algorithm), pre-generates community summaries, and map-reduces over those summaries at query time.
- Which question types benefit most from GraphRAG? == Global sensemaking questions such as "what are the main themes?" and multi-hop relational questions that no single chunk answers.
- What is the main cost of GraphRAG? == Expensive indexing, since LLMs must extract entities and relations and summarize communities, and the graph must be maintained as documents change.
- What is agentic RAG? == An LLM agent decides whether, what, and where to retrieve, issuing follow-up queries and using tools iteratively until it has enough evidence, instead of one fixed retrieval step.
- What are the trade-offs of agentic RAG? == Better handling of multi-hop and ambiguous questions at the cost of higher latency, higher cost, and less predictable behavior.
- What is corrective retrieval in an agentic RAG loop? == Grading retrieved documents for relevance and, if they are insufficient, rewriting the query or querying another source before generating.

## Citations and Grounding

- What does grounding mean in RAG? == Constraining the answer to claims supported by the retrieved context rather than the model's parametric memory.
- How are citations typically implemented in RAG? == Label each chunk with an ID in the prompt, instruct the model to cite IDs per claim, then map IDs back to source documents and locations.
- Why verify citations after generation? == Models can cite a source that does not actually support the claim; checking claim-source entailment catches fabricated or misattributed citations.
- How should a RAG system handle questions the context cannot answer? == Instruct the model to say the sources do not contain the answer instead of guessing, and test this with unanswerable questions.
- Why are retrieved documents a prompt injection risk? == They are untrusted input, so instructions embedded in them can hijack the model (indirect prompt injection); treat context as data and limit tool permissions.

## Failure Modes

- What is the "lost in the middle" effect? == LLMs use information at the beginning and end of long contexts better than in the middle, so relevant passages buried mid-prompt are often ignored.
- How do you mitigate "lost in the middle"? == Pass fewer, well-reranked chunks and place the most relevant ones at the start or end of the context.
- What is a retrieval miss? == The answer exists in the corpus but is not in the retrieved top-k, so the generator cannot use it no matter how capable it is.
- How do you tell a retrieval failure from a generation failure? == Check whether the gold passage is in the retrieved context: if not, retrieval failed; if it is and the answer is still wrong, generation or prompting failed.
- What is a stale index? == An index that lags behind its source data, so the system confidently answers from outdated or deleted documents.
- How do you prevent a stale index? == Incremental re-indexing driven by change detection (content hashes, timestamps), deleting vectors of removed documents, and storing freshness metadata.
- What is a chunk boundary failure? == A fact split across two chunks, or a chunk cut off from the heading or subject it refers to, so no single chunk matches or answers the query.
- What is vocabulary mismatch in retrieval? == Users phrase queries with different terms than the documents (synonyms, acronyms), mitigated by hybrid search and query rewriting or expansion.

## RAG Evaluation

- What is recall@k? == The fraction of a query's relevant documents that appear in the top k retrieved results.
- What is MRR (Mean Reciprocal Rank)? == The mean over queries of 1 / rank of the first relevant result (0 if none is retrieved), rewarding a relevant item near the top.
- What is NDCG? == Normalized Discounted Cumulative Gain: graded relevance discounted by log2(rank + 1), summed, and divided by the ideal ordering's DCG, giving a position-sensitive 0 to 1 score.
- Why is recall@k the key metric for first-stage retrieval? == The reranker and generator can only use what was retrieved, so the first stage should maximize recall and leave precision to later stages.
- What is faithfulness in RAG evaluation? == The degree to which every claim in the answer is supported by the retrieved context; low faithfulness signals hallucination even if the answer happens to be correct.
- What is answer relevancy? == How directly and completely the answer addresses the question, independent of whether its claims are factually correct.
- What is context precision? == Whether the retrieved chunks are relevant to the question, and (in RAGAS) whether relevant chunks are ranked above irrelevant ones.
- What is context recall? == Whether the retrieved context contains all the information needed to produce the reference answer, judged against a ground-truth answer.
- What is a golden dataset for RAG? == A curated set of representative questions with reference answers and/or relevant document IDs, including edge cases and unanswerable questions, used as a fixed regression benchmark.
- What is the main caveat of LLM-as-judge evaluation? == Judges have biases such as position, verbosity, and self-preference, so their scores must be validated against human labels before being trusted.
- Why evaluate retrieval and generation separately? == To attribute failures to the right stage, since improving the generator cannot fix a retrieval miss.
