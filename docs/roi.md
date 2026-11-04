# ROI comparison (optional but preferred by judges)

Two numbers here are measured from the frozen run logs; everything else is labelled as an
assumption so a reader can re-price it. Do not present the assumed rows as measured.

Measured inputs come from `runs/sft2_hard30/trajectories.jsonl` (30 tasks, greedy):
3.57 assistant turns per request, 6257 characters of billed input across the episode, 661
characters of output. At ~3.6 chars/token that is **~1,738 input tokens and ~184 output tokens
per request**. Verify with:

```bash
python - <<'PY'
import json, statistics

rows = [json.loads(l) for l in open("runs/sft2_hard30/trajectories.jsonl", encoding="utf-8")]
conv = [r["metadata"]["conversation"] for r in rows]

def billed(c):
    last = max(i for i, m in enumerate(c) if m["role"] == "assistant")
    return (
        sum(len(m["content"]) for m in c[:last]),
        sum(len(m["content"]) for m in c if m["role"] == "assistant"),
    )

inputs, outputs = zip(*(billed(c) for c in conv))
print(round(statistics.mean(inputs)), round(statistics.mean(outputs)))  # 6257 661
PY
```

## Scenario

| Input | Value | Source |
|---|---:|---|
| Monthly agent requests | 10,000 | assumption, edit to your volume |
| Input tokens / request | 1,738 | measured (see above) |
| Output tokens / request | 184 | measured |
| Turns / request | 3.57 | measured |

## Closed-source API

```
monthly_cost = requests * (input_tokens * input_price + output_tokens * output_price) / 1_000_000
```

Price **assumed** at \$3 / M input and \$15 / M output — replace with the live price of the API you
compare against on submission day, and rerun the arithmetic:

```
10,000 * (1,738 * 3 + 184 * 15) / 1_000_000 = $79.74 / month
```

## Self-hosted Tiny Aya

```
gpu_hours = requests * seconds_per_request / 3600
monthly_cost = gpu_hours * gpu_hour_price
```

GPU price **assumed** at \$0.35/h (T4-class, on-demand). The model is a 3.35B LoRA merge served in
fp16/4-bit, so a single T4-class card serves one request at a time at the latencies below.

| Model | s / request | GPU-hours / month | Cost / month |
|---|---:|---:|---:|
| Base Tiny Aya (measured, 4-bit, 30 tasks / 2428.4 s) | 80.95 | 224.9 | \$78.70 |
| SFT-mix (measured, 4-bit, 30 tasks / 1648.6 s) | 54.95 | 152.6 | \$53.43 |

The eval-24 run writes a third data point: 1212.5 s / 24 = 50.52 s per request, the same
adapter on the in-distribution split.

The honest headline: **at base-model latency, self-hosting is a wash** — \$78.70 against a
\$79.74 API bill, so reliability is the only thing you bought. At tuned latency it is
**0.67× the API bill**. Base needs 7.03 tool calls per task and 16 of 30 episodes never
finish, and GPU time scales with turns; the tuned adapter is cheaper *because* it drops to
3.57 calls, not because the model is smaller.

## Break-even

```
break_even_cost_ratio = self_host_cost_per_request / api_cost_per_request
```

- API: \$0.00797 per request.
- Self-host at base latency: \$0.00787 per request -> 0.99x the API, with 3.33% success. No reason to run it.
- Self-host at tuned latency: \$0.00534 per request -> **0.67x the API, at 96.67% success.**

Break-even monthly volume only matters if there is a fixed cost (dedicated GPU reservation,
monitoring, engineering). With a serverless GPU there is none, so the decision is the per-request
ratio above.

## Honest caveats

- Tiny Aya is less capable than a frontier API. Paired with tool schemas it answers a closed
  ledger question; it is not a general assistant. This is a cost and reliability trade-off, not a
  quality parity claim.
- Self-hosting includes engineering time, monitoring, cold-start latency and an eval harness.
  None of that is priced above. Add it or state that it is excluded.
- Latency was measured on a shared Kaggle T4 with greedy decoding, one process, batch size 1.
  Prompt caching and batching would change both columns.
- LoRA-adapter hosting (what the Space does) is cheaper than merging and re-serving, but adds a
  PEFT dependency; the numbers above price the merged model that was actually evaluated.
