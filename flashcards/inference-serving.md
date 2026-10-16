# Flashcards: LLM Inference & Serving

## Prefill, Decode & Arithmetic Intensity

- What happens in the prefill phase of LLM inference? == All prompt tokens are processed in one parallel forward pass, populating the KV cache and producing the first output token.
- What happens in the decode phase of LLM inference? == Tokens are generated one at a time autoregressively, each step attending to the cached keys and values of all previous tokens.
- Why is prefill typically compute-bound? == It processes many tokens per forward pass, so each weight loaded from memory is reused across many tokens in large matrix-matrix multiplies.
- Why is decode typically memory-bandwidth-bound? == Each step processes only one new token per sequence, so all weights and the KV cache must be read from memory for very little computation.
- What is arithmetic intensity? == The ratio of floating-point operations performed to bytes moved from memory (FLOPs per byte) for a kernel or workload.
- How does arithmetic intensity tell you whether a workload is compute- or memory-bound (roofline model)? == Below the hardware ridge point (peak FLOPs / memory bandwidth) throughput is limited by memory bandwidth; above it, by compute.
- Why does a larger batch make decode more efficient? == Weights are loaded once per step and reused for every sequence in the batch, raising the arithmetic intensity of the weight matmuls.
- What roughly bounds single-sequence decode speed for a dense model? == Memory bandwidth divided by the bytes read per token (all weights plus the KV cache), since every parameter is read once per generated token.
- Why does batching not raise the arithmetic intensity of decode attention over the KV cache? == Each sequence has its own KV cache, so attention reads per-sequence data that is not reused across the batch.

## KV Cache

- What does the KV cache store? == The key and value vectors of every past token at every layer and KV head, so they are not recomputed at each decode step.
- Why are queries not cached during decoding? == Only the current token's query is needed at each step; past queries are never reused.
- What is the formula for KV cache size? == 2 x layers x kv_heads x head_dim x seq_len x bytes_per_element x batch, where the factor 2 accounts for keys and values.
- How does KV cache memory scale with context length and batch size? == Linearly in both, so long contexts and large batches are often limited by KV cache memory rather than by weights.
- What is the KV cache per token for 32 layers, 8 KV heads, head_dim 128, FP16? == 2 x 32 x 8 x 128 x 2 bytes = 131,072 bytes, i.e. 128 KiB per token.
- What is Multi-Query Attention (MQA)? == All query heads share a single key/value head, shrinking the KV cache by a factor equal to the number of query heads, at some quality risk.
- What is Grouped-Query Attention (GQA)? == Query heads are split into groups that each share one KV head, cutting KV cache by num_query_heads / num_kv_heads with quality close to full multi-head attention.
- Why do GQA and MQA also speed up decode, not just save memory? == Less KV data is read from memory per decode step, easing the memory-bandwidth bottleneck of attention.

## Memory Management & Batching

- What problem does PagedAttention (vLLM) solve? == Fragmentation and over-reservation of KV cache memory caused by allocating contiguous max-length buffers per request.
- How does PagedAttention store the KV cache? == In fixed-size blocks allocated on demand and mapped through a per-sequence block table, analogous to OS virtual memory paging.
- How does PagedAttention reduce internal and external fragmentation? == Blocks need not be contiguous, eliminating external fragmentation, and waste per sequence is limited to its last partially filled block.
- What does block sharing in PagedAttention enable? == Parallel samples, beam candidates, or shared prefixes can reference the same physical KV blocks, with copy-on-write when they diverge.
- What is static batching? == A fixed batch runs until every request finishes, so short requests wait for the longest one and new requests wait for the next batch.
- What is continuous (in-flight) batching? == Scheduling at iteration granularity: finished sequences leave and new requests join the running batch at every decode step.
- Why does continuous batching improve throughput over static batching? == Slots freed by finished sequences are refilled immediately, avoiding idle capacity and padding waste caused by variable output lengths.
- What is prefix caching? == Reusing stored KV cache for an exactly matching token prefix (e.g. a system prompt or chat history) across requests, skipping its prefill and reducing TTFT.
- How does SGLang's RadixAttention implement prefix caching? == It keeps KV cache in a radix tree keyed by token sequences, enabling automatic longest-prefix reuse with LRU eviction.
- What is chunked prefill? == Splitting a long prompt's prefill into chunks that are scheduled in the same batches as ongoing decode tokens.
- Why use chunked prefill? == It stops long prefills from stalling running decodes (fewer inter-token latency spikes) and mixes compute-bound and memory-bound work for better utilization.
- What is prefill/decode disaggregation? == Running prefill and decode on separate GPU pools, transferring each request's KV cache from prefill workers to decode workers.
- Why disaggregate prefill and decode, and what does it cost? == It removes interference and lets each pool be sized and parallelized for TTFT or TPOT independently, at the cost of KV cache transfer over fast interconnect and scheduling complexity.
- What can a vLLM-style scheduler do when KV cache memory runs out? == Preempt running requests, either swapping their KV blocks to CPU memory or freeing them and recomputing the prefill later.

## Speculative Decoding

- What is speculative decoding? == A cheap drafter proposes several future tokens and the target model verifies them all in one parallel forward pass, accepting a prefix of them.
- Why does speculative decoding speed up generation? == Decode is memory-bound, so verifying k tokens in one pass costs about as much as generating one, yielding several tokens per target step when drafts are accepted.
- How does speculative sampling keep the output distribution unchanged? == Each draft token is accepted with probability min(1, p_target / p_draft); on rejection a token is resampled from the normalized residual max(0, p_target - p_draft).
- Is speculative decoding lossless? == Yes, with proper rejection sampling the outputs follow exactly the target model's distribution (identical outputs under greedy verification), up to numerical noise.
- What determines the speedup of speculative decoding? == The acceptance rate (draft-target agreement), the number of drafted tokens, and the drafter's cost relative to the target.
- When does speculative decoding help less? == At large batch sizes, where decode is closer to compute-bound so extra verification compute is not free, and when acceptance rates are low.
- What is the Medusa idea? == Add extra decoding heads to the target model that predict several future tokens in parallel, verified with tree attention, removing the separate draft model.
- What is the EAGLE idea? == A lightweight draft head predicts autoregressively at the feature level, using the target model's final hidden states (the features just before the LM head) plus the sampled tokens, which gives high acceptance rates.
- What is n-gram (prompt lookup) speculative decoding? == Drafting tokens by matching n-grams from the prompt or prior context, effective when outputs copy inputs, e.g. code editing or RAG.

## FlashAttention

- What is FlashAttention's core idea? == An IO-aware exact attention kernel that minimizes reads and writes between GPU HBM and on-chip SRAM via tiling and kernel fusion.
- How does FlashAttention avoid materializing the N x N attention matrix? == It processes Q, K, V tiles in SRAM and uses online softmax (running max and sum) to accumulate outputs without writing the score matrix to HBM.
- Is FlashAttention an approximation of attention? == No, it computes exact attention; the speedup comes from less memory traffic, not fewer FLOPs.
- How do FlashAttention's memory and compute scale with sequence length? == Extra memory is linear in sequence length instead of quadratic, while compute stays quadratic.
- What does Flash-Decoding add for inference? == It splits the KV cache along the sequence dimension into chunks processed in parallel and then reduced, keeping the GPU busy during decode with small batches and long contexts.

## Quantization

- What is the difference between weight-only and weight+activation quantization? == Weight-only (e.g. W4A16) stores low-bit weights and dequantizes them for matmuls; weight+activation (e.g. W8A8) quantizes both operands, enabling low-precision tensor-core matmuls.
- Which workload benefits most from weight-only quantization? == Memory-bound decode at small batch, since fewer weight bytes are read per token; compute-bound prefill gains little.
- Why does weight+activation quantization help compute-bound workloads? == Both operands are low precision, so matmuls run on faster INT8 or FP8 tensor cores, raising compute throughput.
- What is FP8 and why is it attractive for activations? == An 8-bit float format (E4M3 favors precision, E5M2 favors range); its exponent handles outliers better than uniform INT8, so it often needs less calibration.
- What is group-wise quantization? == Using a separate scale (and zero-point) per small group of weights, e.g. 128, instead of per tensor or channel, improving low-bit accuracy at small storage overhead.
- Why are LLM activations harder to quantize than weights? == Activations contain large outliers in a few channels, which stretch the quantization range and destroy precision for the remaining values.
- What is GPTQ? == A post-training weight quantization method that quantizes weights column by column, using approximate second-order (Hessian) information to compensate errors in the remaining weights.
- What is AWQ (Activation-aware Weight Quantization)? == A weight-only method that protects salient weight channels, identified by activation magnitude, by scaling them before quantization instead of keeping them in higher precision.
- What is the SmoothQuant idea? == Migrate quantization difficulty from activation outliers to weights with a mathematically equivalent per-channel scaling, making W8A8 INT8 quantization accurate.
- What is GGUF? == The llama.cpp file format bundling quantized weights, tokenizer, and metadata in one file, supporting many quantization types such as Q4_K_M and Q8_0.
- What is KV cache quantization? == Storing keys and values at lower precision (e.g. FP8 or INT8) to cut KV memory and bandwidth, enabling longer contexts or larger batches at small accuracy cost.

## Parallelism

- What is tensor parallelism? == Splitting individual weight matrices (attention heads, MLP rows/columns) across GPUs so each layer is computed jointly, with all-reduce communication in every layer.
- When is tensor parallelism appropriate? == Within a node with fast interconnect such as NVLink, to fit a model too large for one GPU and to reduce per-token latency.
- How does tensor parallelism affect the KV cache? == KV heads are split across GPUs, so each GPU stores only its share of the KV cache, freeing memory for larger batches.
- What is pipeline parallelism? == Assigning consecutive groups of layers to different GPUs and passing activations between stages, communicating only at stage boundaries.
- When is pipeline parallelism appropriate? == Across nodes with slower interconnect, since it communicates less than tensor parallelism; it adds pipeline bubbles and does not reduce single-request latency.
- What is data parallelism in serving? == Running independent model replicas that each handle different requests, scaling throughput with no communication between replicas on the forward path.
- What is expert parallelism? == Distributing MoE experts across GPUs and sending each token to the GPUs holding its selected experts via all-to-all communication.
- When is expert parallelism used? == For large MoE models whose experts dominate memory, typically combined with data or tensor parallelism for the attention layers.

## Latency & Throughput Metrics

- What is TTFT? == Time to first token: from request arrival to the first output token, dominated by queueing and prefill; it drives perceived responsiveness.
- What are TPOT and ITL? == Time per output token (average decode time per token) and inter-token latency (each gap between consecutive streamed tokens).
- How is end-to-end latency approximated from TTFT and TPOT? == E2E latency is approximately TTFT + TPOT x (output_tokens - 1).
- What is throughput in LLM serving? == Total tokens (often output tokens) generated per second across all concurrent requests on a deployment.
- What is goodput? == The throughput of requests that meet their latency SLOs (e.g. TTFT and TPOT targets), measuring useful served load rather than raw tokens per second.
- What is the latency-throughput trade-off of batch size? == Larger batches raise aggregate throughput and cost efficiency but increase per-request TPOT and queueing, until compute or KV cache memory saturates.
- How do per-user and system tokens per second relate? == Per-user decode speed is 1 / TPOT; system throughput sums over concurrent users, and the two move in opposite directions as batch size grows.

## Memory Estimates & MoE

- How do you estimate model weight memory? == params x bytes_per_param, with 4 bytes for FP32, 2 for FP16/BF16, 1 for INT8/FP8, 0.5 for INT4 (plus small scale overhead).
- How much memory do 70B parameters need in FP16 vs INT4? == About 140 GB in FP16 and about 35 GB in INT4, before KV cache and runtime overhead.
- What makes up total GPU memory when serving an LLM? == Weights + KV cache + activations/workspace + framework overhead; memory left after weights sets KV capacity and thus max batch and context.
- What is the difference between total and active parameters in an MoE model? == Total counts all experts and determines memory; active counts shared layers plus the top-k routed experts used per token and determines FLOPs per token.
- Why do MoE models need large batches for efficient inference? == At small batch each expert sees few tokens per weight load, so arithmetic intensity is low; larger batches give each loaded expert more tokens.

## Sampling

- What is greedy decoding? == Always choosing the highest-probability token; deterministic but prone to repetition and bland output.
- What does temperature do in sampling? == It divides logits by T before softmax: T below 1 sharpens the distribution, T above 1 flattens it, and T near 0 approaches greedy.
- What is top-k sampling? == Sampling only from the k most probable tokens after renormalizing their probabilities.
- What is top-p (nucleus) sampling? == Sampling from the smallest set of tokens whose cumulative probability reaches p, so the candidate set adapts to model confidence.
- What is min-p sampling? == Keeping only tokens whose probability is at least min_p times the top token's probability, a cutoff that scales with model confidence.
- How do repetition, presence, and frequency penalties differ? == Repetition penalty rescales logits of previously seen tokens; presence penalty subtracts a constant once a token appeared; frequency penalty subtracts in proportion to its count.
- What is beam search and when is it used? == Keeping the top-b partial sequences by cumulative log-probability each step; useful for short, well-defined outputs like translation, rarely for open-ended chat.
- Why can temperature-0 output differ between runs on a serving engine? == Changing batch composition and non-associative floating-point reductions alter logits slightly, which can flip near-tied argmax choices.

## Structured Output

- What is constrained decoding? == At each step, masking the logits of tokens that would violate a grammar or schema so only valid continuations can be sampled (e.g. Outlines, XGrammar, llama.cpp grammars).
- How are regexes and JSON schemas enforced during decoding? == They are compiled into a finite-state machine or a grammar automaton (pushdown for CFGs), and the current state determines the allowed-token mask.
- Why is aligning tokens with a grammar non-trivial? == Tokens span multiple characters and do not align with grammar symbols, so engines precompute or cache which vocabulary tokens are valid from each automaton state.
- What does constrained decoding guarantee and not guarantee? == It guarantees syntactic validity, not semantic correctness, and overly restrictive constraints can degrade output quality.

## Serving Engines

- What is vLLM? == An open-source high-throughput serving engine that introduced PagedAttention, with continuous batching, prefix caching, quantization, speculative decoding, and an OpenAI-compatible server.
- What is TGI? == Hugging Face's Text Generation Inference server for serving Hugging Face models in production with continuous batching and token streaming.
- What is TensorRT-LLM? == NVIDIA's open-source library for optimized LLM inference on NVIDIA GPUs, with fused kernels, in-flight batching, paged KV cache, and FP8/INT4 quantization support.
- What is SGLang? == A serving engine and frontend language featuring RadixAttention prefix caching and fast constrained decoding, strong for multi-call and agentic workloads with shared prefixes.
- What is llama.cpp? == A C/C++ inference engine that runs GGUF-quantized models efficiently on CPUs, Apple Silicon, and consumer GPUs.
- What is Ollama? == A user-friendly local model runner and manager built on the llama.cpp/GGML ecosystem, exposing a simple CLI and REST API.

## Cost Levers

- How should prompts be structured to benefit from provider prompt caching? == Put static content (system prompt, tool definitions, documents) first and variable content last, so the cached prefix matches and is billed at a discount.
- What do batch APIs trade for lower price? == Latency: requests are processed asynchronously with results returned within a longer window, suiting offline evals and bulk processing.
- Why consider smaller or distilled models for cost? == The smallest model meeting quality targets is far cheaper per token, and fine-tuning or distillation can close the gap on narrow tasks.
- What is model routing (or cascading)? == Sending easy requests to cheap models and escalating hard ones to larger models, cutting average cost while preserving quality.
- Why does limiting output length reduce cost and latency? == Output tokens are usually priced higher and generated sequentially, so concise formats and max_tokens limits cut both spend and end-to-end latency.
