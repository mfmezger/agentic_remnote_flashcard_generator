# Flashcards: LLM Evals

## Eval-Driven Development

- What is eval-driven development for LLM apps? == Defining measurable success criteria and an eval suite first, then iterating on prompts, models, and pipelines against it, analogous to test-driven development.
- What are the three levels of LLM product evaluation in Hamel Husain's framework? == Cheap code-based unit tests on every change, human and model-graded evals on curated sets, and online A/B tests on real users.
- What distinguishes offline from online evals? == Offline evals run on a fixed curated dataset before deployment; online evals measure live production traffic, capturing the real input distribution but with delayed, noisy, or sparse labels.
- Why are generic off-the-shelf metrics like "helpfulness" or "coherence" scores often discouraged? == They rarely map to your application's actual failure modes, so they are hard to act on; application-specific evaluators targeting observed failures are more useful.
- Why should you build one evaluator per failure mode rather than one all-purpose judge? == Narrow evaluators are easier to calibrate, debug, and act on, and an aggregate score hides which specific behavior regressed.
- What is criteria drift in LLM evaluation? == The phenomenon that people refine or change their grading criteria while reviewing outputs, so evaluation criteria cannot be fully specified upfront and must be revised iteratively.

## Golden Datasets and Error Analysis

- What is a golden dataset in LLM evals? == A curated, versioned set of representative inputs, often with reference outputs or labels, used as a stable benchmark to compare system versions.
- Why seed golden datasets from production traces? == Real traces reflect the true input distribution and real failure modes, which synthetic or imagined examples often miss.
- What is error analysis in LLM evals? == Manually reviewing a sample of traces, noting what went wrong, and categorizing failures to decide which evaluators and fixes matter most.
- What is open coding in LLM error analysis? == Writing free-form notes on each reviewed trace describing the first or most important problem observed, without a predefined taxonomy.
- What is axial coding in LLM error analysis? == Grouping open-coded notes into a smaller set of coherent failure-mode categories and counting their frequency to prioritize work.
- When can you stop reviewing traces during error analysis? == At theoretical saturation, when reviewing new traces stops revealing new failure categories.
- How can you generate synthetic eval inputs systematically? == Define dimensions such as persona, feature, and scenario, enumerate combinations of values as tuples, then have an LLM write realistic queries for each tuple.

## Assertions vs Model-Graded Evals

- What are assertion-style (code-based) evals? == Deterministic checks such as regex, substring, JSON schema validation, length limits, or executing generated code, giving cheap, fast, reproducible pass/fail results.
- When are model-graded evals needed instead of assertions? == When the criterion is semantic or subjective, such as tone, faithfulness, or whether an answer addresses the question, and cannot be checked by code.

## LLM-as-a-Judge Design

- What is pointwise LLM-as-a-judge evaluation? == The judge scores a single output in isolation against criteria, giving an absolute grade, with cost scaling linearly in the number of outputs.
- What is pairwise LLM-as-a-judge evaluation? == The judge sees two outputs for the same input and picks the better one or a tie, which is often more reliable for relative comparisons but suffers position bias.
- What is reference-based LLM-as-a-judge evaluation? == The judge compares the output against a gold reference answer, which makes correctness judgments more reliable for factual or well-defined tasks.
- Why prefer binary pass/fail judgments over Likert scales? == Binary labels are more consistent across annotators and judges, easier to calibrate against humans, and more actionable; Likert midpoints are ambiguous and drift.
- What should a good judge rubric contain? == A precise definition of one criterion, explicit pass and fail conditions, and labeled few-shot examples including borderline cases.
- Why ask the judge to reason before giving its verdict? == Because generation is autoregressive, a critique written first conditions the verdict, typically improving accuracy and making judgments debuggable.
- What is G-Eval? == An LLM-judge method that generates evaluation steps via chain-of-thought from the criteria, then scores in form-filling style, weighting candidate scores by their token probabilities.
- What is a panel or jury of LLM judges? == Aggregating verdicts from several judges, ideally from different model families, to reduce single-model bias and variance, sometimes using cheaper smaller models.
- Why should the judge prompt be developed with train, dev, and test splits of human-labeled data? == Train supplies few-shot examples, dev is used to iterate the prompt, and an untouched test split gives an unbiased estimate of judge accuracy.

## Judge Calibration and Error Correction

- How do you validate an LLM judge? == Have humans label a sample of outputs, run the judge on the same items, and measure agreement before trusting the judge's scores.
- Why is raw percent agreement misleading for judge validation? == With imbalanced labels, a judge that always says "pass" can show high agreement while catching no failures.
- What is Cohen's kappa? == Agreement between two raters corrected for chance: kappa = (p_o - p_e) / (1 - p_e), where p_o is observed and p_e is chance-expected agreement.
- What is a judge's true positive rate (TPR), treating pass as the positive class? == Among outputs humans labeled pass, the fraction the judge also labels pass.
- What is a judge's true negative rate (TNR), treating pass as the positive class? == Among outputs humans labeled fail, the fraction the judge also labels fail; it measures how well the judge catches real failures.
- Why report judge TPR and TNR separately rather than accuracy? == They expose asymmetric errors; a judge that misses failures (low TNR) inflates pass rates even when overall accuracy looks good.
- How do you correct an observed judge pass rate for judge error? == Estimate true pass rate as theta = (p_obs + TNR - 1) / (TPR + TNR - 1), clipped to [0, 1] (the Rogan-Gladen estimator).
- When does the judge-error correction formula break down? == When TPR + TNR is close to 1, meaning the judge is near random, the denominator approaches zero and the estimate becomes unstable.

## Judge Biases and Mitigations

- What is position bias in pairwise LLM judges? == A tendency to prefer the answer in a particular position, often the first though the direction varies by model, independent of quality.
- How do you mitigate position bias? == Evaluate each pair in both orders and only count consistent verdicts, treating inconsistent ones as ties.
- What is verbosity bias in LLM judges? == A preference for longer, more detailed answers even when they are not more correct or helpful.
- What is self-preference (self-enhancement) bias? == A judge model's tendency to rate outputs from itself or its own model family more favorably.
- How do you mitigate self-preference bias? == Use a judge from a different model family than the systems being compared, or a panel of diverse judges.
- What judge weakness did the MT-Bench paper highlight besides position, verbosity, and self-enhancement bias? == Limited ability to grade math and reasoning, which can be mitigated with reference-guided judging or having the judge solve the problem first.

## Pairwise Preference Aggregation

- What is the Elo rating update rule? == R_new = R + K * (S - E), where S is the actual result (1, 0.5, 0) and E = 1 / (1 + 10^((R_opp - R) / 400)) is the expected score.
- What is the Bradley-Terry model? == A model where P(i beats j) = 1 / (1 + exp(-(s_i - s_j))), with strengths s fitted by maximum likelihood (logistic regression) over all comparisons at once.
- What advantage does Bradley-Terry have over online Elo for LLM leaderboards? == It is order-independent, uses all battles jointly for more stable estimates, and supports confidence intervals via bootstrapping.
- How does Chatbot Arena (LMArena) collect data? == Users submit a prompt, see responses from two anonymous models side by side, vote for the better one or a tie, and identities are revealed only after voting.
- How does LMArena turn votes into a leaderboard? == It fits a Bradley-Terry model to pairwise votes and reports ratings on an Elo-like scale with bootstrapped confidence intervals.
- What is a common criticism of Arena-style leaderboards regarding style? == Votes reward length, formatting, and tone, so models can climb by style rather than substance; LMArena added style control to adjust for this.
- What criticism of LMArena concerns private testing? == Providers can privately test many model variants and publish only the best, plus get unequal data access, which can inflate rankings (raised in "The Leaderboard Illusion").

## Benchmarks: What They Measure

- What does MMLU measure? == Broad knowledge and problem solving via four-option multiple-choice questions across 57 subjects spanning STEM, humanities, social sciences, and professional fields.
- How does MMLU-Pro differ from MMLU? == It uses ten answer options instead of four, adds more reasoning-heavy questions, and removes trivial or noisy ones, making it harder and less prompt-sensitive.
- What does GPQA measure? == Graduate-level multiple-choice questions in biology, physics, and chemistry written by domain experts and designed to be "Google-proof" for skilled non-experts; Diamond is its hardest high-quality subset.
- What does HumanEval measure? == Functional correctness of Python code generation on 164 hand-written problems given a signature and docstring, verified by unit tests.
- What does SWE-bench measure? == Whether a system can resolve real GitHub issues from popular Python repositories by producing a patch that makes the issue's previously failing tests pass without breaking existing passing tests.
- What is SWE-bench Verified? == A 500-task subset of SWE-bench that human engineers validated to remove underspecified issues and unfair or broken tests.
- What does GSM8K measure? == Multi-step arithmetic reasoning on grade-school math word problems with numeric answers.
- What does the MATH benchmark measure? == Competition-level mathematics problem solving across subjects and five difficulty levels, graded on the final answer.
- What does HellaSwag measure? == Commonsense inference by choosing the most plausible continuation of everyday scenarios, with distractors built via adversarial filtering.
- What does TruthfulQA measure? == Whether models avoid repeating common misconceptions and imitative falsehoods on questions crafted so humans often answer them falsely.
- What does MT-Bench measure? == Multi-turn conversational and instruction-following quality on 80 questions across 8 categories, graded by a strong LLM judge on a 1-10 scale.
- What does IFEval measure? == Instruction following on prompts with verifiable constraints (e.g., word counts, required keywords, no commas) that are checked programmatically.
- What does BFCL measure? == Function-calling accuracy, including choosing the right function, producing correct arguments, parallel and multiple calls, and recognizing when no function is relevant.
- What does tau-bench measure? == Agent reliability in tool-using customer-service tasks with an LLM-simulated user and domain policies, scored by comparing the final database state to the goal state.
- What does ARC-AGI measure? == Fluid abstraction and skill acquisition: inferring a transformation rule from a few input-output grid examples and applying it to a new grid.
- What does Humanity's Last Exam measure? == Frontier expert-level knowledge and reasoning on difficult, partly multimodal questions written by subject experts across many fields, designed to resist saturation.

## Saturation and Contamination

- What is benchmark saturation? == When top models score near the ceiling, so the benchmark no longer discriminates between models or tracks progress.
- What is data contamination in LLM evals? == Benchmark items or close paraphrases appearing in training data, inflating scores through memorization rather than capability.
- How can n-gram overlap detect contamination? == Check for long shared n-grams between benchmark items and the training corpus, which requires access to the training data.
- How can membership-inference signals detect contamination without training data? == Unusually low perplexity or high token probabilities on benchmark items (e.g., Min-K% Prob) suggest the model saw them during training.
- How can perturbation reveal contamination or overfitting? == A large score drop on freshly written or rephrased equivalents of benchmark items, as GSM1k did for GSM8K, indicates memorization.
- What are canary strings? == Unique identifiers embedded in benchmark files so dataset builders can filter them out and researchers can test whether a model has memorized them.
- How do live or time-split benchmarks counter contamination? == They continually add problems released after models' training cutoffs, as LiveCodeBench and LiveBench do.

## Variance, Power, and Prompt Sensitivity

- Why is temperature 0 not fully deterministic for LLM APIs? == Batching, non-associative floating-point arithmetic in GPU kernels, and mixture-of-experts routing can change outputs across calls.
- How should eval runs handle sampling randomness? == Run each item several times, report mean and standard error, and fix seeds and decoding settings where possible.
- Why are paired comparisons preferred when comparing two models? == Scoring both on the same items removes item-difficulty variance, so the per-item difference has lower variance and detects smaller effects.
- What is the approximate per-group sample size to detect a pass-rate difference? == n ≈ (z_alpha/2 + z_beta)^2 * (p1(1 - p1) + p2(1 - p2)) / (p1 - p2)^2, about 7.85 times the variance sum over the squared difference at alpha 0.05 and 80% power.
- Roughly how many items per arm are needed to detect 50% vs 55% at alpha 0.05 and 80% power with independent samples? == About 1,560 per arm, showing that small eval sets cannot detect few-point differences.
- What is prompt sensitivity in evals? == Scores can shift substantially with semantically irrelevant changes like formatting, separators, option order, or few-shot example order.
- How should you mitigate prompt sensitivity when comparing models? == Use identical templates and answer extraction, evaluate across several prompt variants, and report the spread rather than a single number.

## Evaluating RAG and Agents

- Why evaluate retrieval and generation separately in RAG? == It localizes failures: a wrong answer may come from missing context (retrieval) or from ignoring or misusing correct context (generation).
- What does faithfulness (groundedness) measure in RAG eval? == The fraction of claims in the answer that are supported by the retrieved context, capturing hallucination relative to sources.
- What does context recall measure in RAG eval? == Whether the retrieved context contains the information needed for the reference answer; it requires ground-truth answers or relevant passages.
- What is the RAG triad? == Context relevance (query vs retrieved context), groundedness (answer vs context), and answer relevance (answer vs query).
- What is outcome-based agent evaluation? == Judging only whether the final result or end state is correct, e.g., tests pass or a database matches the goal, regardless of path.
- What is trajectory-based agent evaluation? == Judging the sequence of steps, such as tool choices, arguments, and ordering, against expectations or rubrics.
- What is the main risk of strict trajectory matching for agents? == It penalizes valid alternative paths, so it is best used for diagnosis or hard constraints while outcomes determine success.
- What does pass^k measure for agents? == The probability that all k independent trials of a task succeed, estimated per task as C(c, k) / C(n, k), measuring reliability rather than best-case ability.
- How does pass^k differ in behavior from pass@k as k grows? == pass@k rises with k because one success suffices, while pass^k falls with k, exposing inconsistency.

## CI Regression and Online Evaluation

- How are LLM evals typically structured in CI? == Fast deterministic assertions run on every change, larger model-graded suites run on a sample or nightly, and results are compared against a baseline.
- Why should CI eval gates use tolerances instead of exact score matching? == Eval scores are noisy, so gates should flag drops beyond expected run-to-run variance to avoid flaky failures.
- What is an online eval in production? == Running assertions or LLM judges asynchronously on sampled live traffic to monitor quality and detect drift or new failure modes.
- What implicit feedback signals are used to evaluate LLM features online? == Behaviors like accepting, copying, editing, regenerating, or abandoning outputs, which are more abundant than explicit thumbs ratings.
- What is a guardrail metric in an A/B test? == A metric that must not degrade significantly, such as latency, cost, error rate, or safety violations, even if the primary metric improves.
- Why are cost and latency first-class eval dimensions? == A more accurate configuration may be unusable if it is too slow or expensive, so choices should be made on the quality-cost-latency Pareto frontier.
- Why compare cost per successful task rather than cost per call? == A cheaper model that fails more often or needs retries and extra steps can be more expensive per completed outcome.

## Frameworks

- What is OpenAI Evals? == An open-source framework and registry for defining evals from datasets with built-in and model-graded templates, complemented by a hosted Evals API.
- What is Inspect? == The UK AI Security Institute's open-source Python eval framework built from datasets, solvers, and scorers, with support for sandboxed agentic tasks and a log viewer.
- What is promptfoo? == A CLI and config-driven tool that runs a matrix of prompts, providers, and test cases with assertions (deterministic and LLM-rubric), commonly used in CI and red teaming.
- What is DeepEval? == A Python framework offering pytest-style unit tests for LLM outputs with built-in metrics such as G-Eval, answer relevancy, faithfulness, and hallucination.
- What is Ragas? == A Python library focused on evaluating RAG pipelines with metrics like faithfulness, answer relevancy, context precision, and context recall, plus synthetic test set generation.
- What is Pydantic Evals? == A code-first, typed eval library from the Pydantic team where datasets of cases are scored by deterministic or LLM-judge evaluators, with OpenTelemetry tracing integration.
- What does MLflow evaluate provide? == An API that runs a model or agent over a dataset, computes built-in, custom, and LLM-judged metrics, and logs results to MLflow tracking for comparison across runs.
