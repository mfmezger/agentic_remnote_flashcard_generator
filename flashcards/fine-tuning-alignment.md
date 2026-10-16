# Flashcards: Fine-Tuning & Alignment

## Training Stages

- What is pretraining of an LLM? == Self-supervised next-token prediction on a massive general corpus (often trillions of tokens) that gives the model broad knowledge and language ability.
- What is mid-training? == An intermediate stage between pretraining and post-training that continues training on curated, higher-quality or targeted data (e.g., code, math, long context) to shift capabilities before alignment.
- What is post-training? == The stages after pretraining (SFT, preference optimization, RL) that turn a base model into a helpful, instruction-following, safe assistant.
- How does a base model differ from an instruct (chat) model? == A base model only continues text; an instruct model has been post-trained to follow instructions and conversational formats.

## Fine-Tune vs Prompt vs RAG

- When should you prefer prompting over fine-tuning? == When the task can be specified with instructions and a few examples; prompting is cheapest, fastest to iterate, and should be the baseline.
- When is RAG the right tool? == When the model needs large, changing, or private knowledge with citations; retrieval injects facts at inference time without retraining.
- When is fine-tuning the right tool? == To consistently change behavior, style, format, or task skill, to shorten prompts and cut latency, or to distill a large model into a smaller, cheaper one.
- Why is fine-tuning a poor way to add new factual knowledge? == Models absorb new facts slowly and unreliably through fine-tuning and it can increase hallucination; RAG is more reliable and easier to update.

## Supervised Fine-Tuning (SFT)

- What is supervised fine-tuning (SFT)? == Training a pretrained model with next-token cross-entropy loss on curated (prompt, desired response) demonstrations.
- What is instruction tuning? == SFT on a diverse set of tasks phrased as natural-language instructions, improving zero-shot generalization to unseen instructions.
- What is a chat template? == The model-specific format of special tokens and role markers (system, user, assistant) that serializes a conversation into a single token sequence.
- Why must the chat template match between fine-tuning and inference? == A mismatched template puts inputs out of the training distribution, degrading quality, instruction following, and stopping behavior.
- What is loss masking on prompt tokens? == Setting labels of prompt/user tokens to an ignore value (e.g., -100 in PyTorch) so loss is computed only on assistant response tokens.
- Why mask prompt tokens in the SFT loss? == So the model learns to generate responses rather than spending capacity modeling user prompts, which typically improves response quality.
- Why must the end-of-turn or EOS token be included in the trained response? == Otherwise the model never learns when to stop and may keep generating indefinitely.

## Full Fine-Tuning, PEFT, and Memory

- What is full fine-tuning? == Updating all model parameters; it is the most expressive option but needs the most memory and a full model copy per task.
- What is parameter-efficient fine-tuning (PEFT)? == Training only a small set of new or selected parameters while freezing the base model, cutting memory, storage, and compute.
- How many bytes per parameter does full fine-tuning with Adam in mixed precision need? == Roughly 16 bytes: 2 for bf16 weights, 2 for bf16 gradients, 4 for fp32 master weights, and 8 for the two fp32 Adam moment estimates, excluding activations.
- Approximately how much memory does full fine-tuning of a 7B model with Adam need? == About 7B x 16 bytes, roughly 112 GB before activations, so it requires multiple GPUs, sharding (e.g., ZeRO/FSDP), or offloading.
- Why keep fp32 master weights in mixed-precision training? == Small updates can be lost to bf16/fp16 rounding; an fp32 master copy accumulates updates accurately.
- What does gradient checkpointing trade off? == It saves activation memory by recomputing activations during the backward pass, at the cost of extra compute (roughly one additional forward pass).
- What does gradient accumulation do? == It sums gradients over several micro-batches before each optimizer step, simulating a larger effective batch size without extra memory.

## LoRA

- What is the core idea of LoRA? == Freeze the pretrained weight W and learn a low-rank update delta W = BA, so the effective weight becomes W + BA.
- For a weight W of shape d x k, what are the shapes of the LoRA matrices? == B is d x r and A is r x k with rank r much smaller than min(d, k), giving r(d + k) trainable parameters instead of dk.
- How are LoRA matrices initialized? == A gets a random initialization and B is set to zeros, so BA = 0 and training starts exactly from the base model.
- What does the LoRA rank r control? == The capacity of the update: higher r means more trainable parameters and expressiveness; typical values range from about 4 to 64.
- What is LoRA alpha? == A scaling hyperparameter: the update is applied as W + (alpha / r) x BA, which reduces the need to retune the learning rate when changing r.
- Which modules does LoRA typically target? == The original paper adapted only attention projections (mainly q and v); applying it to all linear layers, including the MLP, usually gives better quality.
- What does merging a LoRA adapter mean? == Computing W' = W + (alpha / r) x BA once and replacing W, yielding a standard model with no extra modules.
- Does a merged LoRA model add inference latency? == No; after merging, the architecture and FLOPs are identical to the base model.
- What is the benefit of keeping LoRA adapters unmerged? == Many small task adapters can share one base model and be hot-swapped or batched per request (multi-LoRA serving).
- Why does LoRA reduce training memory so much? == Gradients and optimizer states are kept only for the tiny adapter parameters; the frozen base needs just its weights plus activations.
- How does LoRA compare to full fine-tuning in quality? == LoRA often matches full fine-tuning on narrow tasks but can underperform when much new learning is needed (e.g., new domains), while tending to forget less.

## QLoRA

- What is QLoRA? == LoRA training on top of a frozen base model quantized to 4-bit, backpropagating gradients through the quantized weights into 16-bit LoRA adapters.
- What is NF4? == 4-bit NormalFloat, a data type whose quantization levels are placed for normally distributed weights, which pretrained weights approximately are.
- What is double quantization in QLoRA? == Quantizing the per-block quantization constants themselves, saving about 0.37 bits per parameter.
- What are paged optimizers in QLoRA? == Optimizer states placed in NVIDIA unified memory that pages to CPU RAM, avoiding out-of-memory crashes during memory spikes such as long sequences.
- How are the 4-bit weights used in computation in QLoRA? == They are dequantized on the fly to bf16 for matrix multiplications; storage is 4-bit but compute is 16-bit.
- What is the main tradeoff of QLoRA? == Much lower memory (e.g., fine-tuning a 65B model on a single 48 GB GPU) at the cost of slower training from dequantization overhead.
- What is the caveat when merging a QLoRA adapter? == Merging into 4-bit weights loses precision; typically dequantize the base to 16-bit before merging, or serve the adapter unmerged.

## Other PEFT Methods

- What is prompt tuning? == Learning a small set of continuous soft-prompt embeddings prepended to the input while the whole model stays frozen.
- What is prefix tuning? == Learning continuous prefix vectors prepended to the keys and values at every transformer layer, not just at the input embeddings.
- What are adapter layers (Houlsby-style)? == Small bottleneck MLP modules inserted into each transformer layer and trained with the base frozen; they add some inference latency because they cannot be merged.
- What is DoRA at a high level? == Weight-Decomposed LoRA: splits weights into magnitude and direction, trains the magnitude directly and applies LoRA to the direction; often beats LoRA and is still mergeable.

## Forgetting and Data

- What is catastrophic forgetting? == The loss of previously learned general capabilities when a model is fine-tuned on a narrow distribution.
- How can catastrophic forgetting be mitigated? == Mix in general (replay) data, use lower learning rates and fewer epochs, prefer PEFT such as LoRA, and monitor general benchmarks.
- Data quality vs quantity in fine-tuning? == A small set of high-quality, diverse examples often beats a large noisy set; LIMA got strong instruction following from about 1,000 curated examples.
- What is the superficial alignment hypothesis? == The LIMA claim that a model's knowledge comes from pretraining and alignment mainly teaches the format and style for interacting with users.

## Synthetic Data and Distillation

- What is synthetic data in fine-tuning? == Training data generated by an LLM (e.g., Self-Instruct, teacher responses); cheap and scalable but needs filtering for quality, diversity, and correctness.
- What is knowledge distillation? == Training a smaller student model to imitate the outputs of a larger teacher model.
- What is logit (token-level) distillation? == The student minimizes KL divergence to the teacher's full next-token probability distributions, requiring teacher logits and a shared tokenizer.
- What is sequence-level distillation? == The student is fine-tuned with SFT on text generated by the teacher; it works with API-only teachers since only outputs are needed.

## RLHF

- What are the three stages of the classic RLHF pipeline? == SFT on demonstrations, then a reward model trained on human pairwise preferences, then RL (typically PPO) against that reward with a KL penalty to a reference model.
- Why collect pairwise preferences instead of absolute scores? == Humans judge which of two responses is better more consistently than they assign calibrated absolute ratings.
- What is the Bradley-Terry model for preferences? == P(chosen preferred over rejected) = sigmoid(r(x, y_w) - r(x, y_l)), so the probability depends only on the reward difference.
- What is the reward model training loss? == Minimize -log sigmoid(r(x, y_w) - r(x, y_l)) over preference pairs, pushing chosen rewards above rejected ones.
- What is the typical reward model architecture? == Usually the SFT model with the language-model head replaced by a scalar head that outputs one reward per (prompt, response).
- What objective does the RL stage of RLHF maximize? == E[r(x, y)] - beta x KL(pi || pi_ref): expected reward minus a penalty for drifting from the reference (SFT) policy.
- What is the reference model in RLHF? == A frozen copy of the SFT model used to compute the KL penalty that keeps the policy from drifting.
- Which models are involved in PPO-based RLHF? == Four: the trainable policy, the frozen reference model, the reward model, and the value (critic) model.
- What does PPO's clipping do? == It clips the probability ratio between new and old policies to [1 - eps, 1 + eps], preventing overly large policy updates.
- What is reward hacking? == The policy exploits flaws in an imperfect reward model to score highly without real quality gains, e.g., verbosity, sycophancy, or degenerate text.
- Why include the KL penalty to the reference model? == It limits reward hacking, preserves fluency and diversity, and keeps the policy in the region where the reward model is accurate.
- What is the effect of beta on the KL penalty? == Higher beta keeps the policy closer to the reference (safer, smaller gains); lower beta allows more reward optimization and more hacking risk.
- What is reward overoptimization? == As optimization against a proxy reward model increases, true quality first rises then falls even while the proxy reward keeps climbing (Goodhart's law).

## DPO and Variants

- What is DPO? == Direct Preference Optimization trains the policy directly on preference pairs with a classification-style loss, skipping the explicit reward model and RL loop.
- What is the key insight behind DPO? == The KL-constrained RLHF objective has a closed-form optimal policy, so the reward is implicitly beta x log(pi(y|x) / pi_ref(y|x)) plus a prompt-only term.
- What is the DPO loss? == -log sigmoid(beta x [log(pi(y_w|x) / pi_ref(y_w|x)) - log(pi(y_l|x) / pi_ref(y_l|x))]), raising the chosen log-ratio relative to the rejected one.
- What role does the reference model play in DPO? == The frozen SFT model anchors the log-ratios, acting as the implicit KL constraint that keeps the policy from drifting arbitrarily.
- What does beta control in DPO? == How strongly the policy is held to the reference: higher beta keeps it closer; values around 0.1 are common.
- What are DPO's advantages over PPO-based RLHF? == It is simpler, more stable, and cheaper: no reward model, value model, or online sampling, just supervised-style training on offline pairs.
- What is a key limitation of DPO? == It is offline, learning only from fixed preference data, so it can overfit and lacks the exploration of online RL.
- What is IPO in one line? == Identity Preference Optimization replaces DPO's log-sigmoid with a squared loss on the log-ratio margin to avoid overfitting to preferences.
- What is KTO in one line? == Kahneman-Tversky Optimization learns from unpaired binary desirable/undesirable labels using a prospect-theory-inspired loss.
- What is ORPO in one line? == Odds Ratio Preference Optimization adds an odds-ratio preference penalty to the SFT loss, aligning in a single stage without a reference model.
- What is SimPO in one line? == Simple Preference Optimization uses length-normalized average log-probability as the implicit reward with a target margin, needing no reference model.

## RLAIF and Constitutional AI

- What is RLAIF? == Reinforcement Learning from AI Feedback: an LLM instead of humans produces the preference labels used for reward modeling or preference optimization.
- What is Constitutional AI? == Anthropic's method where the model critiques and revises its own outputs against written principles for SFT, then trains with RL on AI preference labels guided by those principles.

## RLVR, GRPO, and Reasoning

- What is RL with verifiable rewards (RLVR)? == RL where rewards come from programmatic checks, such as matching a math answer or passing unit tests, rather than a learned reward model.
- What is GRPO? == Group Relative Policy Optimization: sample a group of responses per prompt, score them, and use each response's reward relative to the group as its advantage.
- How is the advantage computed in GRPO? == A_i = (r_i - mean(group rewards)) / std(group rewards) over the G responses sampled for the same prompt.
- Why does GRPO not need a value (critic) model? == The group's mean reward serves as the baseline, replacing the learned critic and saving memory and complexity compared with PPO.
- How does GRPO handle the KL constraint? == It adds a KL penalty to the reference model directly to the loss rather than folding it into the reward.
- What are reasoning models? == Models trained, often with RLVR, to produce long chains of thought before answering, improving performance on math, code, and planning tasks.
- What is test-time compute? == Spending more inference compute, via longer reasoning, multiple samples with voting, or search, to improve answer quality.
- What did DeepSeek-R1-Zero demonstrate? == Pure RL with verifiable rewards on a base model, without prior SFT, can elicit long chain-of-thought reasoning and self-verification behaviors.

## Model Merging

- What is model merging? == Combining the weights of multiple fine-tuned models into one model without further training.
- What is task arithmetic? == A task vector is fine-tuned weights minus base weights; adding scaled task vectors to the base adds capabilities, and subtracting removes them.
- What is the main requirement for merging models? == They must share the same architecture and usually the same base checkpoint so their weights remain compatible.

## Evaluation and Overfitting

- How should a fine-tune be evaluated on its target task? == On a held-out set from the target distribution that was never used for training or hyperparameter selection.
- Why run general benchmarks before and after fine-tuning? == To detect regressions such as catastrophic forgetting of general knowledge, instruction following, or safety behavior.
- What baseline must a fine-tune beat to justify itself? == The base or instruct model with good prompting (and RAG where relevant) on the same evaluation set.
- What are classic overfitting signals in fine-tuning? == Training loss keeps falling while validation loss rises, and outputs become memorized, repetitive, or less diverse.

## Hyperparameters

- What is usually the most important fine-tuning hyperparameter? == The learning rate: too high causes instability and forgetting, too low leads to under-training.
- What learning rates are typical for full fine-tuning vs LoRA? == Full fine-tuning often uses around 1e-5 to 2e-5, while LoRA typically uses higher rates around 1e-4 to 2e-4.
- How many epochs are typical for SFT? == Usually 1 to 3; more epochs on small datasets risk overfitting and memorization.
- What is learning-rate warmup? == Gradually increasing the learning rate from near zero over the first steps to avoid destabilizing updates while optimizer statistics are still poorly estimated.
- How does batch size affect fine-tuning? == Larger effective batches give smoother gradient estimates and are often reached via gradient accumulation; the learning rate may need retuning when batch size changes.
