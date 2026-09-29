# ForJev 0.1.0 — Qwen backend for OpenJEV

ForJev turns an **already running Qwen/vLLM HTTP endpoint** into OpenJEV's
typed decision API. It runs a CPU HTTP adapter and loads no LLM, tokenizer
weights, vision tower or learned classification head. The same Qwen instance
continues serving its existing clients. This branch integrates with OpenJEV's
schema validation, authentication, errors and answer formatting.

| URL | Role |
| --- | --- |
| `http://<spark>:8000` | Existing Qwen on pinned Eugr b12x; harness URL stays unchanged |
| `http://<spark>:8001/v1/systemone` | Typed decisions via ForJev |
| `http://<spark>:8001/v1/chat/completions` | Optional transparent chat/stream/tool proxy to Qwen |
| `http://<spark>:8001/v1/models` | ForJev and upstream chat model discovery |
| `http://<spark>:8001/health` | Adapter liveness |
| `http://<spark>:8001/ready` | Adapter plus upstream health |

No Qwen restart, image update, port move or second model download is part of
this installation. Existing hidden-state capture and the older gateway on
8088 are independent and are not used by ForJev.

## Install and serve on DGX Spark

Use the `feature/forjev` branch of this OpenJEV fork in its own directory:

```bash
git clone --branch feature/forjev https://github.com/MhaWay/openjev.git ~/openjev-forjev
```

Python 3.12 on Linux is the tested installer target; `python3-venv` must exist.
Run from the checkout:

```bash
cd ~/openjev-forjev
bash setup-forjev.sh
bash restart-forjev.sh start
```

The installer creates `.venv-forjev`, installs the CPU HTTP requirements,
installs this package with `--no-deps`, and copies `forjev.env.example` to
`.forjev.env` only if absent. It deliberately does not install upstream's
Transformers/model dependencies. Do not use upstream's default Docker Compose
or DiffusionGemma entrypoint for this deployment.

Defaults in `.forjev.env`:

```sh
OPENJEV_BACKEND=forjev
OPENJEV_UPSTREAM=http://127.0.0.1:8000
OPENJEV_UPSTREAM_MODEL=qwen3.8-flash-next-a5b
OPENJEV_HOST=0.0.0.0
OPENJEV_PORT=8001
OPENJEV_MAX_INFLIGHT=2
FORJEV_MAX_CHOICES=20
```

`OPENJEV_UPSTREAM` is the server root, **without `/v1`**. Set the model to an
actual ID returned by upstream `/v1/models`. For an authenticated upstream,
set `FORJEV_UPSTREAM_API_KEY`. Frontend authentication uses the independent
`OPENJEV_API_KEY` and `OPENJEV_ORIGIN_SECRET`; these frontend credentials are
not passed to Qwen. Leave keys empty for the existing VPN/local setup.

```bash
bash restart-forjev.sh status
bash restart-forjev.sh probe
bash restart-forjev.sh restart
bash restart-forjev.sh stop
```

This Linux controller uses a lock, PID plus process creation identity, and
a detached process with logs under `.forjev-run/`. It only signals the
ForJev process it started from this checkout. It never invokes Docker or
changes/stops Qwen. Start verifies the upstream model, token IDs and complete
candidate logprobs before launching, then checks `/ready`, `/v1/models` and
a real SystemOne response. Failed starts stop only the new adapter. Restart
checks the upstream before stopping the old adapter; it does not automatically
roll back source changes. Fix the reported error and run start again.

The detached adapter survives terminal/harness disconnection. It does not
automatically start after a machine reboot. The harness stays on Qwen :8000,
so no reconnect listener or model restart is necessary.

Manual foreground launch, after exporting the settings above:

```bash
.venv-forjev/bin/python -m openjev
```

## How the decision works

For each nontrivial question, ForJev formats state, question and options into
a prompt, assigns each option a distinct single-token label, and optionally
includes the supplied image parts. It requests one output token with thinking
disabled and reads the **next-token logprobs of every candidate label**.
It normalizes these with `exp(logprob - max) / sum(exp(...))` and delegates
choice/score/noul formatting to OpenJEV.

The full prompt prefill still runs. There is one upstream chat request per
question and `max_tokens=1`, not a full generated explanation. A one-option
choice can be returned without inference. OpenJEV's `output_tokens: 0`
denotes the decision-only wire contract, not zero upstream compute or zero
provider-billed output tokens. Input usage sums actual upstream prompt tokens.

Probabilities are relative to the provided options, not calibrated real-world
certainty. OpenJEV's entropy-based confidence is preserved. Label order,
prompt wording and quantization can affect results. This backend is not a
trained JevK5 checkpoint or diffusion inference implementation.

## API and token IDs required from Qwen/vLLM

The adapter currently requires:

1. `GET /health` and `GET /v1/models` for its launcher preflight.
2. `POST /tokenize` accepting `model`, `prompt`, `add_special_tokens:false`,
   returning numeric `tokens`. This resolves the labels in the served tokenizer.
3. `POST /v1/chat/completions` supporting `logprobs:true`,
   `logprob_token_ids:[...]`, and `return_tokens_as_token_ids:true`.
   The answer must include all requested candidates in the first output
   position's `top_logprobs`, with `token:"token_id:<integer>"` and `logprob`.
4. For vision, OpenAI-style `image_url` content and a multimodal model/template.
   The Qwen template must honor `chat_template_kwargs:{"enable_thinking":false}`.

These are per-request options. No ForJev worker extension, pooling runner,
dev RPC or hidden-state patch is required. The previously tested pinned b12x
build supports this route for small candidate sets; verify this new integration
live before treating it as production-ready. Do not upgrade a working image
merely to install this adapter.

Inspect the actual IDs and test all configured candidates:

```bash
bash restart-forjev.sh probe
```

The JSON includes `labels`, e.g. `{"A":32,"B":33}` if those are what the live
tokenizer reports, and the probabilities returned by a real request. IDs are
discovered and cached in process; they are never hardcoded or substituted from
another Qwen tokenizer. Restart ForJev after changing the upstream model/tokenizer.

To test an image explicitly, export the environment and run:

```bash
set -a; source .forjev.env; set +a; .venv-forjev/bin/python -m openjev.forjev_probe --choices 20 --image /path/to/image.jpg
```

Default support is 20 choices. You may configure `FORJEV_MAX_CHOICES` up to
255, but only if enough distinct single-token labels exist and vLLM accepts
that many `logprob_token_ids`. Some builds cap this at 128; `--max-logprobs`
and the explicit-token-ID cap are not necessarily the same setting. Inspect
the exact serving build and rerun the probe at the requested count. Missing
candidate logprobs cause an error; ForJev does not silently invent probabilities.

The adapter cannot turn an arbitrary cloud chat endpoint into this exact
readout. A provider needs these capabilities (plus authentication support);
ordinary top-k logprobs do not guarantee that every option is present.

## Requests

Model ID: `forjev-qwen-next`. OpenJEV SDK aliases `jev-latest` and
`jev-preview` also select this backend. The upstream Qwen chat ID is separate.

Text-only state: omit `images` or send `images:null`. The state is still sent
to Qwen; absence of an image does not skip inference.

```bash
curl -fsS http://127.0.0.1:8001/v1/systemone -H 'Content-Type: application/json' -d '{"model":"forjev-qwen-next","state":{"health":4,"threat":"zombie"},"questions":{"action":{"type":"choice","instructions":"Choose the safest action.","criteria":{"retreat":"Move away from the zombie","approach":"Move toward the zombie"}}}}'
```

Images are passed as OpenJEV top-level `images`, for example:

```json
{
  "model": "forjev-qwen-next",
  "state": "Classify this image.",
  "images": ["data:image/jpeg;base64,<actual base64 bytes>"],
  "questions": {
    "letter": {
      "type": "choice",
      "instructions": "Which uppercase letter is visible?",
      "criteria": {"B": "The letter B", "D": "The letter D"}
    }
  }
}
```

`choice`, `score` and `noul`, structured states/instructions and multiple
questions use the existing OpenJEV schema. Images are optional; no frame/session
upload from the older prototype is required. Unsupported `steps>1`,
`samples>1`, `think>0` and `sequential:true` are explicitly rejected.

## Version, compatibility and validation

ForJev backend version: **0.1.0**. The upstream OpenJEV package version is
retained separately. Derived from OpenJEV commit
`a0ddd7d928298eccef2c17153b00b5636b6d996a` and the earlier
`MhaWay/Qwen3.8-Flash-Next-Int4-FAST:feature/forjev` prototype.

Current target: Qwen3.8-Flash-Next served by the existing pinned Eugr b12x
stack. Future Qwen4 compatibility is a goal, not an already tested guarantee;
the HTTP contract, template, labels and quality must be checked with each model.

Offline tests cover the native OpenJEV API, state/image handling, typed
answers, complete candidate probabilities, labels beyond 52, unsupported
options, chat streaming/auth, and adapter start/stop against a fake upstream.
They do not establish accuracy, calibration, latency, 255-choice availability
or compatibility with a real future model.

```bash
.venv-forjev/bin/python -m pip install pytest
.venv-forjev/bin/python -m pytest -q tests/test_forjev.py tests/test_forjev_service.py
```

Credits and licensing: OpenJEV by razorback16 and contributors, Apache-2.0;
ForJev integration uses the same repository license. Existing model licenses
and API provider terms still apply. The other OpenJEV backends are retained.
