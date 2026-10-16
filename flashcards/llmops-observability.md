# Flashcards: LLMOps & Observability

## Observability Signals

- What are the three classic observability signals? == Traces (request-scoped causal paths), metrics (aggregated numeric time series), and logs (discrete timestamped event records).
- What question are metrics best suited to answer? == "Is something wrong, and how much?" — they are cheap, aggregated, and ideal for dashboards and alerting, but lose per-request detail.
- What question are traces best suited to answer? == "Where and why did this specific request go wrong or slow down?" — they show the causal tree of operations and their timings.
- How are logs correlated with traces? == By stamping each log record with the active trace ID and span ID, so you can jump from a log line to its full trace.
- What is a metric exemplar? == A sample trace ID attached to a metric data point (e.g. a histogram bucket), linking an aggregate spike to a concrete example trace.

## OpenTelemetry Core Concepts

- What is a span in OpenTelemetry? == A single timed operation with a name, start/end timestamps, attributes, events, status, and a parent span ID, forming one node of a trace.
- What is a trace in OpenTelemetry? == A tree of spans sharing one trace ID that represents the end-to-end path of a single request or workflow.
- What are span attributes? == Key-value pairs on a span describing the operation (e.g. HTTP method, model name), used for filtering and aggregation.
- What are span events? == Timestamped annotations within a span's lifetime, such as a recorded exception or a streaming milestone.
- What are the possible span status values in OpenTelemetry? == Unset (default), Ok, and Error; Error marks the operation as failed.
- What is an OpenTelemetry Resource? == Immutable attributes describing the entity producing telemetry, such as service.name, service version, and host or deployment environment.
- What is context propagation? == Passing the active trace context across function, thread, and process boundaries so child spans attach to the correct parent and trace.
- How is trace context propagated across services over HTTP by default? == Via W3C Trace Context headers (traceparent and optionally tracestate), injected by the caller and extracted by the callee using a propagator.
- What are the four fields of a W3C traceparent header? == Version, trace ID (32 hex chars), parent span ID (16 hex chars), and trace flags (e.g. 01 means sampled), separated by hyphens.
- What is OpenTelemetry Baggage? == Application-defined key-value pairs propagated alongside trace context across services; it is not automatically added to spans as attributes.
- What is an exporter in OpenTelemetry? == The component that sends finished telemetry to a backend, e.g. via OTLP to a Collector or vendor, or to the console for debugging.
- What is the OpenTelemetry Collector? == A vendor-neutral service that receives, processes (batching, filtering, redaction, sampling), and exports telemetry through configurable pipelines.
- What are the default ports for OTLP over gRPC and OTLP over HTTP? == gRPC uses port 4317; HTTP uses port 4318.
- When might you choose OTLP/HTTP over OTLP/gRPC? == When proxies, load balancers, or firewalls handle plain HTTP more easily than HTTP/2 gRPC, or in environments like browsers where gRPC is impractical.
- How does BatchSpanProcessor differ from SimpleSpanProcessor? == BatchSpanProcessor buffers ended spans and exports them asynchronously in batches (use in production); SimpleSpanProcessor exports each span synchronously on end (use for debugging/tests).
- What is a risk of using BatchSpanProcessor in short-lived processes? == Buffered spans can be lost on exit unless you flush or shut down the tracer provider before the process terminates.

## Sampling

- What is head-based sampling? == The keep/drop decision is made at trace start, before outcomes are known; it is cheap but can miss rare errors or slow requests.
- What is tail-based sampling? == The decision is made after the whole trace is collected (typically in a Collector), allowing policies like "keep all errors and slow traces".
- What is a downside of tail-based sampling? == All spans of a trace must be buffered in one place until the decision, which costs memory and requires routing a trace's spans to the same collector instance.
- What does a TraceIdRatioBased sampler do? == Samples a configured fraction of traces deterministically from the trace ID, so all services using the same ratio agree on the decision.
- What does a ParentBased sampler do? == Follows the parent span's sampling decision when one exists and delegates to another sampler (e.g. ratio) only for root spans, keeping traces complete.

## GenAI Semantic Conventions

- What are the OpenTelemetry GenAI semantic conventions? == Standardized gen_ai.* attribute names and span shapes for LLM calls, tools, and agents; they are still in Development (experimental) status and evolving, so names may change.
- What does the gen_ai.operation.name attribute capture? == The type of GenAI operation, such as chat, embeddings, tool execution, or agent invocation.
- What is the difference between gen_ai.request.model and gen_ai.response.model? == The request model is the name you asked for (possibly an alias); the response model is the exact model version the provider reports having used.
- Which GenAI attributes record token usage? == gen_ai.usage.input_tokens and gen_ai.usage.output_tokens, enabling per-span token and cost accounting.
- What span types do the GenAI conventions describe for agentic apps? == Model call spans (e.g. chat), tool execution spans, and agent invocation spans that parent the model and tool spans of an agent run.
- How do the GenAI conventions treat prompt and completion content? == Capturing message content is opt-in because it can contain sensitive data and is large; metadata like model and tokens is captured by default.

## What to Trace in LLM Apps

- What should a well-instrumented LLM call span include? == Model and parameters, prompt/template version, input and output (if allowed), token counts, latency, finish reason, errors, and estimated cost.
- Why trace retrieval steps in a RAG app separately? == To see the query, retrieved document IDs and scores, and latency, so you can tell whether bad answers stem from retrieval or generation.
- Why trace each tool call as its own span? == To capture tool name, arguments, result, errors, and latency, making agent loops and failing tools debuggable.
- How should user feedback be connected to observability data? == Attach it to the trace (by trace ID) so ratings can be analyzed alongside the exact prompt, context, and model output.
- What metadata enables slicing traces by user or session without exposing identity? == Pseudonymous user and session IDs (e.g. hashed) plus tags like feature, tenant, and environment.

## Privacy and PII

- Why are LLM traces a PII risk? == Prompts, retrieved context, and completions often contain personal or confidential data that gets copied into observability backends with broad access.
- What are common mitigations for PII in LLM traces? == Redact or mask before export (SDK or Collector), make content capture opt-in, restrict access, set retention limits, and support deletion requests.

## Tracing Tools

- What is MLflow Tracing? == MLflow's OpenTelemetry-compatible tracing for GenAI apps, using autologging for popular libraries and a decorator or context manager for custom spans.
- What is Langfuse? == An open-source, self-hostable LLM engineering platform combining tracing, prompt management, evaluations, datasets, and user feedback.
- What is LangSmith? == LangChain's commercial platform for tracing, evaluating, and monitoring LLM apps; it works with or without LangChain.
- What is Arize Phoenix? == An open-source LLM observability and evaluation tool built on OpenTelemetry, using the OpenInference conventions for LLM spans.
- What is Pydantic Logfire? == An observability platform built on OpenTelemetry by the Pydantic team, with integrations for Pydantic AI, LLM SDKs, and web frameworks.
- What is the main advantage of OpenTelemetry-based LLM tracing tools? == Vendor neutrality: the same instrumentation can export to different backends and join LLM spans with the rest of your distributed traces.

## Prometheus Metrics

- What is a Prometheus counter? == A monotonically increasing value (reset only on restart), queried with rate() or increase(); e.g. total requests or total tokens.
- What is a Prometheus gauge? == A value that can go up and down, such as in-flight requests, queue depth, or GPU memory usage.
- What is a Prometheus histogram? == Observations counted into configurable cumulative buckets plus a sum and count; quantiles are computed server-side and can be aggregated across instances.
- How does a Prometheus summary differ from a histogram? == A summary computes quantiles client-side, which cannot be meaningfully aggregated across instances; histograms are generally preferred for latency.
- What are key Prometheus metric naming conventions? == Use snake_case, base units in the name (e.g. _seconds, _bytes), and the _total suffix for counters.
- What is the label cardinality problem in Prometheus? == Every unique label-value combination creates a new time series, so unbounded labels explode memory and query cost.
- Why should user IDs or request IDs never be Prometheus labels? == They are unbounded and create a time series per value; put them in traces or logs instead and keep labels low-cardinality (model, route, status).

## Monitoring Methods and SLOs

- What is the RED method? == For request-driven services, monitor Rate (requests/s), Errors (failed requests), and Duration (latency distribution).
- What is the USE method? == For resources (CPU, GPU, memory, queues), monitor Utilization, Saturation, and Errors.
- What is an SLI? == A Service Level Indicator: a measured ratio of good events to valid events, e.g. the fraction of requests answered successfully under 2 seconds.
- What is an SLO? == A Service Level Objective: a target for an SLI over a time window, e.g. 99.5% of requests succeed over 28 days.
- How does an SLA differ from an SLO? == An SLA is an external contract with consequences (e.g. credits) if missed; SLOs are internal targets usually set stricter than the SLA.
- What is an error budget? == The allowed unreliability implied by an SLO (1 minus the target); while budget remains you can ship changes, and when it is exhausted you prioritize reliability.
- What is burn-rate alerting? == Alerting on how fast the error budget is being consumed relative to the rate that would exactly exhaust it over the SLO window (burn rate 1); multi-window burn-rate alerts catch both sudden outages and slow degradations with fewer false alarms.

## LLM Production Metrics

- What is TTFT? == Time To First Token: latency from sending the request to receiving the first streamed token, the main driver of perceived responsiveness.
- What does output tokens per second measure? == Generation throughput after the first token; together with TTFT and output length it determines total response latency.
- Why track refusal rate separately from error rate? == Refusals are successful HTTP responses where the model declines to help, so they hide from error metrics but signal prompt, policy, or model-change problems.
- Why track cost per request rather than only total spend? == It normalizes for traffic, exposes regressions from longer prompts or model changes, and enables per-feature and per-tenant cost attribution.
- Why monitor cache hit rate in an LLM app? == It directly affects cost and latency; a sudden drop often means a prompt change broke the cacheable prefix or cache keys.
- Why monitor latency percentiles (p95/p99) rather than averages? == Averages hide the tail; LLM latency is highly variable with output length, and tail requests dominate user frustration and timeouts.

## Prompt and Model Management

- Why manage prompts outside application code? == Versioned prompts in a registry can be reviewed, rolled back, and A/B tested independently of code deploys, with each trace linked to its prompt version.
- What is a prompt version label or alias (e.g. "production")? == A movable pointer to a specific immutable prompt version, so promotion and rollback change the pointer rather than the code.
- Why pin a dated model snapshot instead of a floating alias? == Floating aliases can be repointed by the provider to new versions, silently changing behavior; pinned snapshots keep outputs reproducible.
- How should you handle a provider's model deprecation notice? == Track retirement dates, run your regression eval suite on the replacement model, adjust prompts, then migrate via canary before the cutoff.

## Gateways and Resilience

- What is an LLM gateway or router? == A proxy layer that unifies provider APIs and centralizes auth, routing, fallbacks, retries, rate limiting, caching, cost tracking, and logging.
- What is a model fallback? == Automatically retrying a failed or rate-limited request on an alternative model or provider, ideally one validated by evals for that use case.
- Why use exponential backoff with jitter for retries? == Growing delays reduce pressure on an overloaded service, and random jitter prevents many clients from retrying in synchronized waves.
- Which LLM API errors are generally worth retrying? == Rate limits (429), server errors (5xx), and timeouts or connection errors; client errors like 400 or 401 should not be retried.
- How should a client handle an HTTP 429 response? == Back off and retry later, honoring the Retry-After header if present, and reduce concurrency or request volume if 429s persist.
- What two dimensions do LLM provider rate limits commonly use? == Requests per minute and tokens per minute, so large prompts can hit limits even at low request counts.
- Why set explicit timeouts on LLM calls? == Without them, hung requests tie up workers and blow latency budgets; streaming calls also benefit from an idle timeout between chunks.
- What is a circuit breaker in an LLM gateway? == A mechanism that stops sending requests to a failing provider for a cooldown period and routes to fallbacks, avoiding wasted retries.

## Caching and Cost Control

- What is exact (response) caching for LLMs? == Returning a stored response when the full request key (model, parameters, messages) matches exactly; safe but with low hit rates for free-form input.
- What is semantic caching? == Returning a cached response when a new query's embedding is within a similarity threshold of a past query; higher hit rate but risk of wrong answers.
- What is provider prompt caching? == The provider reuses computation for a repeated prompt prefix across requests, lowering input-token cost and latency without caching the response itself.
- How should prompts be structured to benefit from provider prompt caching? == Put stable content (system prompt, tool definitions, static documents) first and variable content (user input) last so the prefix stays identical.
- What are the main levers for LLM cost control? == Route easy tasks to cheaper models, cap max output tokens, trim context, use caching and batch APIs, and enforce per-user or per-tenant budgets.

## Feedback and Quality Monitoring

- What is explicit user feedback? == Direct signals users give, such as thumbs up/down, star ratings, or written comments on a response.
- What are examples of implicit feedback signals? == Copying or accepting output, regenerating, editing the answer, rephrasing the question, abandoning the session, or escalating to a human.
- What is an annotation queue? == A workflow where sampled or flagged traces are routed to human reviewers for labeling, producing ground truth for evals and datasets.
- What are online evals? == Automated quality scoring (heuristics or LLM-as-judge) run asynchronously on a sample of production traces, usually reference-free since no ground truth exists.
- Why run online evals on sampled rather than all traffic? == LLM-judge evaluation adds cost and load; a representative sample is enough to track quality trends and catch regressions.
- What is input drift in an LLM application? == A shift in the distribution of user queries (topics, languages, length) away from what prompts and evals were designed for.
- How can you detect quality drift in production? == Track online eval scores, feedback rates, refusal rates, and clusters of input embeddings over time, alerting on significant changes.
- How do production traces feed back into offline evaluation? == Failing or interesting traces are curated into eval datasets, so future prompt and model changes are tested against real cases.

## Deployment Strategies

- What is a canary deployment for a prompt or model change? == Routing a small percentage of live traffic to the new version, comparing its metrics with the baseline, and expanding or rolling back.
- What is a shadow deployment? == Sending copies of live requests to the new version in parallel without showing its responses to users, to compare outputs, latency, and cost safely.
- What is an A/B test for prompts or models? == A randomized split of users between variants, measuring quality and business metrics until differences are statistically significant.
- Why use sticky assignment in LLM A/B tests? == Keeping each user on one variant avoids inconsistent experiences and contamination between groups across a conversation or session.

## MLflow Concepts

- What is an MLflow experiment? == A named container grouping related runs (or traces) so they can be compared and searched together.
- What is an MLflow run? == A single execution of code that records parameters, metrics, artifacts, and tags.
- How do MLflow params, metrics, and artifacts differ? == Params are input configuration values logged once, metrics are numeric results that can be logged over steps, and artifacts are output files like models or plots.
- What is the MLflow Model Registry? == A central store of registered models with numbered versions, aliases (e.g. champion), tags, and lineage back to the runs that produced them.
- What are assessments on MLflow traces? == Feedback (quality judgments from humans, LLM judges, or code) and expectations (ground-truth labels) attached to traces or spans for evaluation.
