# Flashcards: Agents & Tool Use

## Agents vs Workflows

- In Anthropic's "Building effective agents", what distinguishes a workflow from an agent? == Workflows orchestrate LLMs and tools through predefined code paths; agents let the LLM dynamically direct its own process and tool usage.
- What does Anthropic call the shared building block of both workflows and agents? == The augmented LLM: a model enhanced with retrieval, tools, and memory that it can use on its own initiative.
- What is Anthropic's general advice on when to build an agent? == Start with the simplest solution, often a single optimized LLM call, and add workflows or agents only when the extra complexity measurably improves results.
- What is the prompt chaining pattern? == Decomposing a task into a fixed sequence of LLM calls where each processes the previous output, optionally with programmatic checks (gates) between steps.
- What is the routing pattern? == Classifying an input and directing it to a specialized downstream prompt, tool, or model, e.g., sending easy queries to a cheaper model.
- What are the two variants of the parallelization pattern? == Sectioning (split a task into independent subtasks run in parallel) and voting (run the same task several times for diverse outputs or consensus).
- What is the orchestrator-workers pattern? == A central LLM dynamically breaks a task into subtasks, delegates them to worker LLMs, and synthesizes their results.
- How does orchestrator-workers differ from parallelization? == In parallelization the subtasks are fixed in advance by code; in orchestrator-workers the orchestrator decides the subtasks at runtime based on the input.
- What is the evaluator-optimizer pattern? == One LLM generates a response while another evaluates it and gives feedback in a loop, best when clear evaluation criteria exist and iteration measurably helps.
- What is the basic loop of an autonomous agent? == The model repeatedly chooses an action (usually a tool call), receives environment feedback, and continues until the task is done or a stopping condition such as max steps is hit.
- What is the key trade-off of agents versus workflows? == Agents offer flexibility for open-ended tasks but cost more, add latency, and risk compounding errors; workflows are more predictable for well-defined tasks.

## Reasoning & Planning Patterns

- What is ReAct? == A prompting paradigm that interleaves reasoning traces with actions (tool calls) and observations, letting the model plan, act, and update based on results.
- What is the canonical ReAct step format? == Thought, then Action (tool plus input), then Observation (tool result), repeated until a final answer is produced.
- What advantage does ReAct have over chain-of-thought alone? == Grounding reasoning in external observations reduces the hallucination and error propagation that purely internal reasoning suffers from.
- What is Reflexion? == An approach where an agent verbally reflects on feedback from failed attempts and stores those reflections in memory to improve later trials, without weight updates.
- Why is Reflexion described as "verbal reinforcement learning"? == The learning signal becomes natural-language self-reflection kept in an episodic memory buffer instead of gradient updates to model weights.
- What is plan-and-execute? == An architecture where a planner first produces a multi-step plan and an executor carries out the steps, optionally replanning after results come in.
- What are the benefits of plan-and-execute over step-by-step ReAct? == Up-front planning improves long-horizon coherence and allows a cheaper model to execute steps, with fewer calls to the expensive planner.
- What is the main weakness of plan-and-execute? == A static plan goes stale when observations contradict its assumptions, so it needs a replanning step to adapt.

## Function and Tool Calling

- What does an LLM actually do during tool calling? == It emits a structured request naming a tool and its arguments; the application executes the tool and returns the result, the model never runs code itself.
- How are tools typically described to a model? == With a name, a natural-language description, and a JSON Schema for the input parameters, all placed into the model's context.
- Why do tool descriptions and parameter names matter so much? == They are the model's main information for deciding which tool to call and how, so clear, unambiguous descriptions are effectively prompt engineering for tool use.
- What does the tool choice setting control? == Whether the model may decide (auto), must call some tool (required/any), must call a specific named tool, or must not call tools (none).
- What are parallel tool calls? == The model emitting multiple independent tool calls in one turn so the application can execute them concurrently and return all results together.
- How are tool results returned to the model? == As tool result messages linked to the originating calls by their IDs, appended to the conversation so the next model turn can use them.
- How should tool errors be reported to the model? == As tool results with informative error messages, so the model can recover, retry with corrected arguments, or choose another approach.
- Why should tool outputs be kept concise? == Tool results consume context tokens; verbose outputs crowd out useful information and degrade reasoning, so truncate, paginate, or summarize them.
- What is Anthropic's agent-computer interface (ACI) advice? == Invest in tool design as much as prompt design: clear docs, examples, and formats hard to misuse (poka-yoke), e.g., requiring absolute instead of relative file paths.

## Structured Outputs

- What are structured outputs? == A feature that makes model output conform to a supplied JSON Schema, guaranteeing parseable, schema-valid responses.
- How does JSON mode differ from schema-enforced structured outputs? == JSON mode only guarantees syntactically valid JSON; structured outputs additionally guarantee conformance to a specific schema.
- What is constrained decoding? == At each generation step, masking out tokens that would violate a grammar or schema so only valid continuations can be sampled.
- How is a JSON Schema typically enforced during constrained decoding? == It is compiled into a grammar or finite-state machine that tracks which next tokens are valid and masks the logits of all others.
- What is a known quality risk of constrained decoding, and a common mitigation? == Forcing a rigid format can degrade reasoning; place a reasoning field before answer fields so the answer, generated left to right, can condition on it.
- Do structured outputs guarantee correct values? == No; they guarantee format validity only, so values can still be wrong or hallucinated and semantic validation is still needed.

## Model Context Protocol (MCP)

- What is the Model Context Protocol (MCP)? == An open protocol, introduced by Anthropic in late 2024, that standardizes how AI applications connect to external tools, data, and prompts.
- What integration problem does MCP address? == It turns the M times N problem (every app integrating every tool) into M plus N by giving apps and tool providers one shared protocol.
- What is an MCP host? == The AI application the user interacts with, such as an IDE or chat app, which creates and manages one or more MCP clients.
- What is an MCP client? == A component within the host that maintains a dedicated one-to-one connection with a single MCP server.
- What is an MCP server? == A program that exposes capabilities (tools, resources, and prompts) to clients over MCP.
- What message format does MCP use? == JSON-RPC 2.0 messages.
- What are MCP tools? == Executable functions exposed by a server that the model can choose to invoke; they are model-controlled.
- What are MCP resources? == Read-only data identified by URIs, such as files or records, that the application can supply as context; they are application-controlled.
- What are MCP prompts? == Reusable prompt templates exposed by servers, typically chosen explicitly by the user (e.g., as slash commands); they are user-controlled.
- What is the stdio transport in MCP? == The host launches the server as a local subprocess and exchanges JSON-RPC messages over its standard input and standard output.
- What is the Streamable HTTP transport in MCP? == Clients send JSON-RPC over HTTP POST to one endpoint, and servers reply with JSON or stream messages via Server-Sent Events; it supports remote servers.
- Which earlier MCP transport did Streamable HTTP replace? == The original HTTP+SSE transport, which used separate endpoints for client POSTs and a long-lived server event stream.
- Why is MCP a security concern? == Servers expose powerful tools and untrusted content; tool descriptions or results can carry prompt injections, so servers must be trusted and risky actions approved.

## Agent Memory

- What is short-term memory in an LLM agent? == The information in the current context window (conversation history, tool results, scratchpad), lost once it falls out of context or the session ends.
- What is long-term memory in an LLM agent? == Information persisted outside the context window, e.g., in databases, vector stores, or files, and retrieved into context across sessions.
- What is episodic memory for agents? == Records of specific past experiences or interactions, often reused as few-shot examples or lessons learned.
- What is semantic memory for agents? == Stored facts and knowledge about the world, the user, or the domain, independent of when they were learned.
- What is procedural memory for agents? == Knowledge of how to perform tasks, encoded in model weights, code, and system prompts or instructions that can be updated over time.
- What are the two main timing strategies for writing long-term memory? == In the hot path, where the agent saves memories during the conversation, or in the background, where a separate process extracts them asynchronously.
- What is a key risk of long-term agent memory? == Stale, wrong, or injected memories persist and get retrieved later, so memory needs curation, updating, and provenance.

## Context Engineering

- What is context engineering? == Curating exactly which tokens enter the model's context at each step (instructions, tools, history, retrieved data) to maximize task performance.
- Why treat the context window as a budget? == Every token competes for finite capacity and attention; low-value content raises cost and latency and can degrade performance.
- What is compaction in agents? == Summarizing or pruning a long conversation history as it nears the context limit, then continuing with the condensed version.
- What is a risk of aggressive compaction? == Losing details that later matter; good compaction keeps decisions, open issues, and key facts while dropping redundant raw tool output.
- What is "lost in the middle"? == The finding (Liu et al., 2023) that LLMs use information at the beginning and end of long contexts better than information in the middle.
- What is context rot? == Degradation in a model's ability to accurately recall and use information as context length grows, even well within the nominal window.
- What is just-in-time context retrieval? == Keeping lightweight references such as file paths, queries, or IDs and letting the agent load data via tools when needed, instead of pre-loading everything.
- What is structured note-taking in long-running agent tasks? == The agent writes persistent notes, such as a progress or TODO file, outside the context and re-reads them, preserving state across compactions.
- Why does a stable prompt prefix matter for agent cost? == Prompt caching reuses computation for identical prefixes, so keeping system prompts and tool definitions stable and appending new content cuts cost and latency.

## Multi-Agent Systems

- What is the supervisor multi-agent pattern? == A central agent routes work to specialized sub-agents and decides next steps based on their results.
- What is a handoff in multi-agent systems? == One agent transfers control of the conversation, plus relevant context, to another agent better suited for the next part of the task.
- Why use subagents for context isolation? == Each subagent explores in its own fresh context and returns a condensed result, keeping the lead agent's context clean and focused.
- What tasks benefit most from multi-agent parallelism? == Breadth-first, parallelizable tasks like broad research, where subagents can explore independent directions simultaneously.
- What is the main cost trade-off of multi-agent systems? == They use far more tokens than single agents (Anthropic reported about 15 times a chat for its research system), so they only pay off on high-value tasks.
- Why are multi-agent systems poorly suited to tightly coupled tasks like most coding? == Subagents lack shared context and make conflicting implicit decisions, so coordination overhead and inconsistency outweigh parallelism gains.
- What is a common failure when delegating to subagents? == Vague task descriptions cause duplicated work, gaps, or misinterpretation; delegation needs clear objectives, output formats, and boundaries.

## Human-in-the-Loop

- What is human-in-the-loop (HITL) for agents? == Pausing the agent at defined points so a human can approve, edit, or reject actions or provide input before it continues.
- Which agent actions typically require human approval? == Irreversible, high-impact, or externally visible actions, e.g., deleting data, spending money, sending messages, deploying, or pushing code.
- What enables an agent to pause for approval and resume later? == Persisting agent state via checkpointing, so execution can be interrupted and resumed after human input.
- What is approval fatigue? == When users see too many prompts they rubber-stamp them, eroding safety; approvals should be reserved for genuinely risky actions.

## Failure Modes

- What is agent looping? == Repeating the same or similar actions without progress, often due to unclear feedback; mitigated with step limits and repetition detection.
- What is tool hallucination? == The model calls nonexistent tools, invents arguments, or fabricates tool results instead of actually calling the tool.
- Why do errors compound over long horizons? == If each step succeeds with probability p, success over n dependent steps is roughly p to the power n, so small per-step error rates cause large end-to-end failure.
- What is reward hacking in agents? == Optimizing the measurable proxy instead of the intended goal, e.g., special-casing or deleting tests to make them pass.
- What is prompt injection in agentic contexts? == Malicious instructions in tool results, web pages, or files that the agent follows as commands, potentially causing harmful actions or data exfiltration.
- What is the "lethal trifecta" for agent data exfiltration? == Simon Willison's term for combining access to private data, exposure to untrusted content, and the ability to communicate externally.
- What is premature task completion? == The agent declares success without verifying, e.g., claiming tests pass without running them; mitigated by requiring verification evidence.

## Agent Evaluation

- What is task success rate in agent evaluation? == The fraction of tasks whose final outcome meets the success criterion, typically checked by tests, environment state checks, or graders.
- What is trajectory evaluation? == Assessing the sequence of steps, tool calls, and reasoning an agent took, not just its final answer, to judge efficiency, correctness, and safety.
- What is a pitfall of strict trajectory matching? == Agents can reach correct results via many valid paths, so rigid matching against a reference trajectory penalizes legitimate alternatives.
- What does pass@k measure? == The probability that at least one of k independent attempts succeeds, capturing capability when retries are allowed.
- What does pass^k measure? == The probability that all k independent attempts succeed, capturing reliability and consistency; it was introduced with the tau-bench benchmark.
- How do pass@k and pass^k behave as k grows? == pass@k rises toward 1 while pass^k falls, exposing inconsistency that single-trial success rates hide.
- If per-trial success is p and trials are independent, what is pass^k? == p to the power k; e.g., 80% per-trial success gives about 51% pass^3.
- Why track cost and latency per task in agent evals? == Agents vary widely in tokens, tool calls, and wall time; an accurate but slow or expensive agent may be unusable, so report them alongside success.
- What is LLM-as-judge in agent evaluation? == Using a model with a rubric to grade outputs or trajectories where programmatic checks are impractical; it should be validated against human labels.
- Why must agent eval environments be reset between trials? == Leftover state lets trials influence each other, producing correlated and misleading results.

## Sandboxing and Permissions

- Why run coding agents in a sandbox? == To contain the impact of mistakes or prompt-injected actions by limiting filesystem, network, and process access to a bounded environment.
- What are the two key isolation dimensions for coding-agent sandboxes? == Filesystem isolation (restrict reads and writes to allowed directories) and network isolation (restrict outbound connections to approved hosts).
- Why is network isolation important even when the filesystem is restricted? == Without it, a compromised agent can exfiltrate secrets or source code, or download malicious payloads.
- What is the principle of least privilege for agents? == Grant only the minimum tools, credentials, and permissions the task needs, scoped narrowly and ideally time-limited.
- What are common permission modes for coding agents? == Read-only or plan mode, ask-before-acting approval, auto-approve within an allowlist, and fully autonomous mode inside an isolated sandbox.
- Why prefer allowlists over denylists for agent commands? == Denylists are easily bypassed via equivalent commands or obfuscation, while allowlists permit only known-safe operations.
- What technologies commonly provide sandboxes for coding agents? == Containers, microVMs such as Firecracker, or OS-level sandboxing such as macOS Seatbelt or Linux bubblewrap and seccomp.
- Why keep secrets out of a coding agent's environment? == Anything the agent can read can leak via outputs, logs, or injected commands, so use scoped, short-lived credentials or credential proxies instead.
