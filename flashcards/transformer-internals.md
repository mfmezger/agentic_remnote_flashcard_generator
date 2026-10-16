# Flashcards: Transformer & LLM Internals

## Attention Fundamentals

- What is the scaled dot-product attention formula? == Attention(Q, K, V) = softmax(Q K^T / sqrt(d_k)) V, where d_k is the query/key dimension.
- Why are attention scores divided by sqrt(d_k)? == With unit-variance components, q·k has variance d_k; scaling restores unit variance so softmax doesn't saturate into near one-hot outputs with vanishing gradients.
- What does the query vector represent intuitively? == What the current token is looking for; it is compared against all keys to decide where to attend.
- What does the key vector represent intuitively? == What each token offers for matching; its dot product with a query determines how much attention that token receives.
- What does the value vector represent intuitively? == The content a token contributes; each attention output is an attention-weighted sum of value vectors.
- How are Q, K, and V computed in self-attention? == By multiplying the same hidden states X by three learned projection matrices: Q = X W_Q, K = X W_K, V = X W_V.
- What does each row of softmax(Q K^T / sqrt(d_k)) represent? == A probability distribution over key positions (weights summing to 1) giving how much one query token attends to each token.
- What is causal masking? == Setting attention scores for future positions (j > i) to minus infinity before softmax, so each token attends only to itself and earlier tokens.
- Why is causal masking needed in decoder LMs? == It prevents leakage from future tokens, so training on all positions in parallel matches left-to-right autoregressive generation at inference.
- What is the difference between self-attention and cross-attention? == In self-attention Q, K, V come from the same sequence; in cross-attention queries come from one sequence (e.g., decoder) and keys/values from another (e.g., encoder output).
- Where is cross-attention used in the original encoder-decoder Transformer? == In each decoder block: decoder states form the queries, attending over the encoder's final outputs as keys and values.
- What is the complexity of standard attention in sequence length n? == O(n^2) time and memory, because every token is scored against every other token, producing an n x n score matrix per head.
- Is unmasked self-attention sensitive to token order on its own? == No; without positional information it is permutation-equivariant (shuffling inputs just shuffles outputs), so order must be injected explicitly.
- What is FlashAttention? == An exact, IO-aware attention kernel that tiles computation in fast on-chip SRAM and never materializes the n x n matrix, making memory linear in n while compute stays O(n^2).

## Multi-Head Attention, MQA & GQA

- What is multi-head attention? == Running h attention operations in parallel on separate learned projections (head dim typically d_model / h), concatenating their outputs, and applying an output projection W_O.
- Why use multiple attention heads instead of one? == Different heads can attend to different positions and relation types in different subspaces simultaneously, at roughly the cost of one full-width head.
- What is the KV cache? == Stored keys and values of previous tokens during autoregressive decoding, so each new token computes only its own Q, K, V instead of reprocessing the prefix.
- How does KV cache memory scale? == Proportional to 2 (K and V) x layers x KV heads x head_dim x sequence length x batch size x bytes per element.
- What is Multi-Query Attention (MQA)? == All query heads share a single key head and a single value head, drastically shrinking the KV cache and memory bandwidth during decoding.
- What is Grouped-Query Attention (GQA)? == Query heads are split into groups that each share one K/V head; it interpolates between multi-head (groups = heads) and multi-query (one group).
- Why do many modern LLMs (e.g., Llama 2 70B, Llama 3) use GQA? == It cuts KV cache size and decode bandwidth almost as much as MQA while keeping quality close to full multi-head attention.

## Architecture Families

- What characterizes an encoder-only transformer? == Bidirectional self-attention over the full input producing contextual embeddings; suited to classification, embeddings, and token labeling (e.g., BERT).
- What objective was BERT pretrained with? == Masked language modeling (predicting randomly masked tokens from both left and right context), plus next-sentence prediction in the original paper.
- What characterizes a decoder-only transformer? == A stack of causally masked self-attention blocks trained to predict the next token; used for generation (e.g., GPT family, Llama).
- What characterizes an encoder-decoder transformer? == A bidirectional encoder reads the input and an autoregressive decoder generates output via cross-attention; suits seq2seq tasks like translation (e.g., original Transformer, T5, BART).
- What pretraining objective does T5 use? == Span corruption: random contiguous spans are replaced by sentinel tokens and the decoder generates the missing spans, within a unified text-to-text framing.
- Why are most frontier LLMs decoder-only? == A single next-token objective scales simply on unlabeled text, natively supports generation and in-context learning, and pairs efficiently with KV caching.

## The Transformer Block

- What are the two main sublayers of a transformer block? == Multi-head self-attention, which mixes information across positions, and a position-wise feed-forward network (MLP), which transforms each token independently.
- What does the classic transformer FFN look like? == Two linear layers with a nonlinearity between them, expanding d_model to a hidden size (classically 4 x d_model) and projecting back.
- What is a residual connection in a transformer block? == Adding a sublayer's input to its output, x + f(x), which eases gradient flow through deep stacks.
- What is the residual stream? == The per-token hidden vector running through the network that each attention and MLP sublayer reads from and adds its output to.
- What does LayerNorm compute? == Per token, it subtracts the mean and divides by the standard deviation across features, then applies a learned scale and bias.
- How does RMSNorm differ from LayerNorm? == It skips mean-centering and bias, dividing only by the root mean square of the features before a learned scale; cheaper and comparably effective (used in Llama).
- What is post-norm vs pre-norm? == Post-norm (original Transformer) normalizes after the residual add: Norm(x + f(x)); pre-norm normalizes the sublayer input: x + f(Norm(x)).
- Why do most modern LLMs use pre-norm? == It keeps a clean identity path through the residual stream, making deep models train more stably and less dependent on careful learning-rate warmup.
- Why do pre-norm models apply a final norm after the last block? == The residual stream itself is never normalized inside the blocks, so it is normalized once before the LM head.
- What is GELU? == Gaussian Error Linear Unit: GELU(x) = x · Phi(x), where Phi is the standard normal CDF; a smooth ReLU-like activation used in BERT and GPT-2.
- What is SwiGLU? == A gated FFN: (Swish(x W1) elementwise-multiplied by (x W3)) W2, with Swish(z) = z · sigmoid(z); used in PaLM and Llama.
- Why is the SwiGLU hidden size often about 8/3 x d_model instead of 4 x d_model? == SwiGLU uses three weight matrices instead of two, so the hidden size is shrunk to keep parameters and compute roughly equal to a standard 4x FFN.

## Positional Information

- What are sinusoidal positional encodings? == Fixed vectors of sines and cosines at geometrically spaced frequencies, added to token embeddings in the original Transformer.
- What are learned absolute positional embeddings? == A trainable vector per position index added to token embeddings (e.g., BERT, GPT-2); positions beyond the trained maximum are unsupported.
- What is RoPE (Rotary Position Embedding)? == Rotates pairs of query and key dimensions by angles proportional to position, so the q·k score depends only on the relative offset between tokens.
- Where is RoPE applied in the model? == To queries and keys inside every attention layer, not to values and not to the input embeddings.
- What is ALiBi? == Attention with Linear Biases: adds a head-specific penalty proportional to query-key distance to attention scores, using no positional embeddings; designed to extrapolate to longer inputs.
- What is position interpolation for extending RoPE context? == Scaling position indices down (by trained length / target length) so longer sequences fall within the trained rotation range, usually followed by brief fine-tuning.
- What is NTK-aware RoPE scaling? == Increasing the RoPE base instead of scaling all positions uniformly, so high-frequency dimensions (local detail) are preserved while low-frequency ones are stretched.
- What is YaRN, conceptually? == A RoPE context-extension method that treats frequency bands differently (keeps high frequencies, interpolates low ones) and adjusts attention temperature, extending context with little fine-tuning.

## Tokenization & Embeddings

- What is Byte-Pair Encoding (BPE)? == Starting from characters or bytes, repeatedly merge the most frequent adjacent pair into a new token until the target vocabulary size; merges are replayed in order when encoding.
- What is byte-level BPE? == BPE over raw UTF-8 bytes with a 256-symbol base vocabulary, so any string can be encoded without unknown tokens (used from GPT-2 onward).
- What is WordPiece? == A subword algorithm (used by BERT) that picks merges by training-data likelihood gain rather than raw pair frequency; continuation pieces are marked with "##".
- What is the Unigram LM tokenizer? == Starts from a large candidate vocabulary and iteratively prunes tokens whose removal least hurts corpus likelihood; encoding picks the most probable segmentation.
- What is SentencePiece? == A library that trains BPE or Unigram tokenizers directly on raw text, treating whitespace as an ordinary symbol, so no language-specific pre-tokenization is needed (used by T5, Llama 1/2).
- Why do token counts differ across languages for the same content? == Vocabularies are fitted to training data dominated by some languages (often English), so underrepresented languages and scripts split into more tokens, raising cost and shrinking effective context.
- Why do LLMs often struggle with arithmetic and spelling tasks? == Tokenization chunks numbers and words into inconsistent multi-character pieces, so the model doesn't directly see individual digits or letters.
- How have tokenizers mitigated digit quirks? == By splitting numbers into single digits (e.g., Llama 1/2, PaLM) or groups of at most three digits (e.g., GPT-4's cl100k), giving more consistent number representations.
- What is the token embedding matrix? == A learned lookup table of shape vocab_size x d_model mapping each token ID to the dense vector fed into the first block.
- What is the LM head? == The final linear layer projecting each position's last hidden state (size d_model) to vocab_size scores used for next-token prediction.
- What are logits in an LLM? == The unnormalized per-vocabulary scores output by the LM head; softmax turns them into a next-token probability distribution.
- What are tied input/output embeddings? == Reusing the token embedding matrix (transposed) as the LM head weights, saving vocab_size x d_model parameters; common in smaller models.

## Training Objective

- What is the next-token prediction objective? == Maximize the likelihood of each token given all preceding tokens, i.e., minimize the cross-entropy loss -log p(true next token), with p from softmax over logits, averaged over all predicted positions.
- How does perplexity relate to cross-entropy loss? == Perplexity = exp(mean cross-entropy in nats); lower perplexity means the model is less surprised by the data.
- What is teacher forcing? == Feeding ground-truth previous tokens as inputs during training rather than the model's own predictions, which enables parallel training over all positions.
- What is exposure bias? == The train-inference mismatch caused by teacher forcing: at inference the model conditions on its own possibly erroneous outputs, which it never saw in training.
- Why can a decoder-only model compute the loss for every position in one forward pass? == Causal masking makes each position's prediction depend only on earlier tokens, so all n next-token predictions are valid simultaneously.

## Mixture of Experts

- What is a Mixture of Experts (MoE) layer in an LLM? == A replacement for the dense FFN consisting of many expert FFNs plus a router; each token is processed by only a few selected experts.
- What does the MoE router do? == A small learned layer scores each token against all experts, selects the top-k, and combines their outputs weighted by the router's gate values.
- What is top-k routing? == Each token is sent only to its k highest-scoring experts (commonly k = 1 or 2), keeping per-token compute small regardless of the total number of experts.
- Why is a load-balancing auxiliary loss used in MoE training? == Without it the router tends to collapse onto a few experts; the loss pushes toward roughly uniform token distribution, improving utilization and stability.
- What is expert capacity in MoE? == A cap on tokens each expert processes per batch; overflow tokens are dropped from that layer and pass through via the residual connection.
- What is the difference between total and active parameters in MoE? == Total counts all experts and sets memory needs; active counts parameters used per token and sets compute (e.g., Mixtral 8x7B: ~47B total, ~13B active).
- What is the main serving trade-off of MoE models? == Per-token FLOPs resemble a much smaller dense model, but all experts must stay in memory, so hardware memory scales with total parameters.

## Scaling Laws & Emergence

- What do neural scaling laws state? == Pretraining loss falls predictably as a power law in parameters, training tokens, and compute, when the other factors aren't bottlenecks.
- What did Kaplan et al. (2020) conclude about allocating more compute? == Grow model size much faster than data, favoring very large models trained on relatively few tokens.
- What did Chinchilla (Hoffmann et al., 2022) conclude? == For compute-optimal training, parameters and training tokens should scale roughly equally; many earlier models (e.g., GPT-3, Gopher) were undertrained.
- What is the Chinchilla rule of thumb for compute-optimal data? == About 20 training tokens per parameter (e.g., Chinchilla's 70B parameters on ~1.4T tokens).
- What is the standard approximation for training compute? == C ≈ 6 · N · D FLOPs, with N parameters and D training tokens (about 2ND forward plus 4ND backward).
- Why do many modern models train far beyond 20 tokens per parameter? == Chinchilla optimizes training compute only; overtraining a smaller model costs more training but makes inference cheaper, which dominates for widely deployed models.
- What are emergent abilities in LLMs? == Capabilities that seem to appear abruptly above some scale while being near random in smaller models, such as multi-step arithmetic.
- What is the main argument that emergent abilities are a mirage? == Schaeffer et al. (2023) showed many stem from discontinuous metrics like exact match; with continuous metrics, performance often improves smoothly with scale.
- What is in-context learning? == Performing a new task from instructions or demonstrations in the prompt without any weight updates, popularized by GPT-3's few-shot results.
- What is the difference between zero-shot and few-shot prompting? == Zero-shot gives only a task description; few-shot also includes several input-output demonstrations in the prompt.

## Decoding & Sampling

- What is greedy decoding? == Picking the single most probable token at each step; deterministic but prone to repetition and may miss more probable full sequences.
- What is beam search? == Keeping the B highest-scoring partial sequences at each step and returning the best finished one; common in translation but tends to bland, repetitive text in open-ended generation.
- How does temperature affect sampling? == Logits are divided by T before softmax; T below 1 sharpens the distribution, T above 1 flattens it, and T approaching 0 approaches greedy decoding.
- What is top-k sampling? == Restricting sampling to the k most probable tokens, renormalizing their probabilities, then sampling.
- What is top-p (nucleus) sampling? == Sampling from the smallest set of tokens whose cumulative probability reaches p, so the candidate set adapts to the model's confidence.
- What is min-p sampling? == Keeping only tokens with probability at least min_p times the top token's probability, so the cutoff scales with the model's confidence.
- Why is top-k considered less adaptive than top-p? == A fixed k admits many unlikely tokens when the distribution is peaked and excludes plausible ones when it is flat.

## Model Sizing & Efficient Alternatives

- What is the approximate parameter count of a dense transformer? == About 12 · n_layers · d_model^2, excluding embeddings, assuming standard multi-head attention and a 4x FFN.
- Where does the 12 · d_model^2 per-layer estimate come from? == Attention has Q, K, V, O projections (4 · d^2) and the FFN has two d x 4d matrices (8 · d^2); biases and norms are negligible.
- How well does 12 · L · d^2 match GPT-3? == With 96 layers and d_model = 12288 it gives about 174B, close to the reported 175B parameters.
- What is sliding-window attention? == Each token attends only to the previous W tokens, reducing cost from O(n^2) to O(n · W) (e.g., Mistral 7B with W = 4096).
- How can stacked sliding-window layers use context beyond the window? == Each layer extends the receptive field by up to W tokens, so after L layers information can flow from roughly L x W tokens back.
- What inference benefit does sliding-window attention give? == The KV cache can be a fixed-size rolling buffer of W entries, bounding memory regardless of sequence length.
- What are state space models (SSMs) in the LLM context? == Sequence models using a linear recurrence over a fixed-size hidden state, giving linear time in sequence length and constant memory during generation (no KV cache that grows with context).
- What is Mamba's key innovation over earlier SSMs? == Selective state spaces: SSM parameters depend on the input, letting the model choose what to keep or forget, computed with a hardware-aware parallel scan.
- What is the main trade-off of SSMs like Mamba versus attention? == The fixed-size state compresses history, so exact recall and copying from long context is weaker; hybrid models interleave attention and SSM layers.
