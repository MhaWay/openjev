# Numeric decision scoring for ForJev

ForJev can request a numeric next-token readout from the resident vLLM model.
Prompt construction, images, tokenizer discovery, labels and typed answers
remain unchanged. No learned head or second model is required.

## Implementation status

| `FORJEV_SCORING` | Upstream path | Status |
| --- | --- | --- |
| `chat_logprobs` | `/v1/chat/completions` | Existing default, backwards compatible |
| `engine_scores` | `/v1/decision_scores` | Numeric bridge implemented; live runtime integration required |
| `prefill_scores` | `/v1/decision_scores` | Contract implemented; native runner provider still required |

The engine bridge reads AsyncLLM results without `_create_chat_logprobs`.
It still schedules one generated token and uses the internal logprob collector,
sampler and scheduler. It cannot repair missing upstream probabilities and does
not promise MTP bypass or latency savings. Missing/misaligned data fails closed.

`prefill_scores` requires genuine prefill-only execution and never downgrades
to generation, argmax or fabricated confidence. Without a native provider the
route returns HTTP 501 before rendering/inference. An environment setting alone
does not implement that provider.

## Runtime integration

Install this package inside the existing vLLM runtime and call
`openjev.vllm_scores.install_routes(app)` on its existing FastAPI app before
serving. Do not construct another engine or load another model. The exact
insertion point must be checked against the installed runtime sources; this
branch does not automatically deploy, patch or restart any running container.

The `/v1` route inherits the app's authentication middleware. It validates the
model and uses vLLM's existing multimodal chat renderer. The initial renderer
supports only the resident base model, no LoRA, tools or arbitrary template
overrides, with `enable_thinking=false`. Compatibility code covers legacy
`_preprocess_chat` and newer `render_chat_request`, pending checks on B12X.
The engine bridge requires raw_logprobs mode and sufficient max_logprobs.

Set `FORJEV_SCORING=engine_scores` after installing the route, then run the
existing ForJev capability probe. It reports the chosen scoring mode. Existing
SystemOne clients and TypeSafe answer shapes need no changes.

## Numeric contract

The request contains `model`, `messages`,
`chat_template_kwargs: {"enable_thinking": false}`, `candidate_token_ids`
(unique IDs discovered through the served tokenizer), and `require_prefill`.

| Response field | Meaning |
| --- | --- |
| `schema` | `forjev.decision_scores.v1` |
| `execution` | `engine_logprobs` or `prefill_logits` |
| `score_type` | `raw_logprobs` or `raw_logits`; processed scores are rejected |
| `token_ids`, `scores` | All requested IDs with their raw values |
| `probabilities` | Stable softmax over the candidate set |
| `candidate_mass` | Full vocabulary probability mass, or null without its normalizer |
| `generated_tokens` | 1 for the engine bridge; 0 for genuine prefill execution |
| `usage.prompt_tokens` | Rendered prompt token count |

The gateway reorders scores by ID and computes softmax itself. Missing,
duplicate, nonfinite or incorrectly typed results are errors. These are model
next-token probabilities, not calibrated probabilities of action success.

## Native provider requirements

`install_routes(app, prefill_score=provider)` accepts an async provider with
the same signature as `engine_scores`. It must integrate into the resident
scheduler and return the verified prefill_logits contract. A production
provider must preserve all of the following:

1. Identical prompt/template and multimodal preprocessing.
2. Target-model logits at the position predicting the first answer token,
   before sampler penalties, temperature, truncation and masking.
3. Request identity and position through batch compaction, async execution and
   chunked prefill; no global last-logits slot.
4. No sampling, MTP drafting or speculative verification for scoring requests;
   normal planner generation retains its existing path.
5. Cancellation, cache ownership and resource cleanup. Cache hits may still
   need the final position recomputed to obtain a readout.
6. GPU gathering of candidate columns; optional full-vocabulary logsumexp
   for candidate_mass. Use the actual loaded lm_head and its quantization.

Selected-row lm_head projection is a later optimization requiring equivalence
tests; pooled embeddings and hidden states before final normalization are not
a substitute for its actual input.

## Verification

CPU tests cover logit/logprob normalization equivalence, full vocabulary mass,
reordered/missing candidates, execution metadata, typed answers, image forwarding,
route validation, request identity, cancellation and backwards compatibility.

Run `python -m pytest tests/test_decision_scores.py tests/test_vllm_scores.py tests/test_forjev.py tests/test_forjev_service.py -q`.

Live acceptance requires the actual B12X build: compare distributions with
successful legacy calls and exercise concurrent planner traffic, batch sizes,
chunked prefill, prefix cache hits/misses and cancellation. Measure latency
on hardware; CPU tests establish no speedup.
