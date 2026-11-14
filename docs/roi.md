# ROI — self-hosted Tiny Aya vs a closed API on the same workload

The judged question is not "is a 3.35B model cheaper", it is "what does this workload cost per task, before and after fine-tuning, each way". Both sides below run the *same* recorded traffic: token counts are measured from the stored transcripts with the base model's own tokenizer, and the self-hosted side uses the wall time each eval actually took on a single T4.

## Measured workload (single T4, greedy, one request at a time)

| Run | Tasks | Success | Calls/task | Input tok/task | Output tok/task | Wall s | Tok/s end-to-end |
|---|---:|---:|---:|---:|---:|---:|---:|
| `sft4_hard90` | 90 | 93.3% | 3.38 | 6,193 | 213 | 4,518 | 127.6 |
| `base_hard90` | 90 | 3.3% | 7.24 | 13,391 | 1,027 | 6,435 | 201.7 |
| `sft4_eval` | 24 | 100.0% | 3.29 | 6,069 | 211 | 1,179 | 127.8 |

`hard257` is not a row here because that run stored wall time and calls but not token counts:
15,848.5 s for 257 tasks is **61.7 s/task at 3.72 calls**, longer per task than hard-90's 50.2 s.
The dollar columns stay anchored to the 90-task traffic, where the token counts were measured.

## Cost per 1,000 tasks (T4 at $0.53/GPU-hour)

| Run | Self-host GPU-hours | Self-host $ | GPT-6 Luna (cheapest frontier tier) $ | GPT-4o-mini $ | GPT-6.1 Sol (mid tier) $ |
|---|---:|---:|---:|---:|---:|
| `sft4_hard90` | 13.94 h | $7.39 | $0.73 | $1.06 | $14.52 |
| `base_hard90` | 19.86 h | $10.53 | $1.85 | $2.62 | $37.05 |
| `sft4_eval` | 13.65 h | $7.23 | $0.71 | $1.04 | $14.24 |

## At 10,000 tasks/month

| Run | Self-host GPU-hours/month | Self-host $/month | GPT-6 Luna (cheapest frontier tier) $/month | GPT-4o-mini $/month | GPT-6.1 Sol (mid tier) $/month |
|---|---:|---:|---:|---:|---:|
| `sft4_hard90` | 139.44 h | $73.90 | $7.26 | $10.57 | $145 |
| `base_hard90` | 198.60 h | $105 | $18.52 | $26.25 | $370 |
| `sft4_eval` | 136.49 h | $72.34 | $7.12 | $10.37 | $142 |

## Cost per 1,000 solved tasks

| Run | Solved / 1,000 | Self-host $ | GPT-6 Luna (cheapest frontier tier) $ | GPT-4o-mini $ | GPT-6.1 Sol (mid tier) $ |
|---|---:|---:|---:|---:|---:|
| `sft4_hard90` | 933 | $7.92 | $0.78 | $1.13 | $15.56 |
| `base_hard90` | 33 | $316 | $55.63 | $78.82 | $1,113 |
| `sft4_eval` | 1,000 | $7.23 | $0.71 | $1.04 | $14.24 |

## What the table says

- **Cost per solved task is the number that matters.** Untuned, the base solves 3.3% of these questions, so its bill buys a correct ledger only every 30th request: $55.63 per 1,000 solved on the cheapest closed tier, against $0.78 for the adapter at 93.3% — 72x less. Self-hosted the same comparison is $316 against $7.92. The gap is the accuracy, not the price list.
- The token saving is the mechanism: 213 generated tokens per task against 1,027 untuned, and 6,193 billed input tokens against 13,391, because the adapter finishes in 3.38 calls instead of 7.24 and stops when the ledger is settled instead of narrating a wrong split.
- Self-hosted on one T4 at $0.53/h the same 1,000 tuned tasks cost $7.39 (13.94 GPU-hours), against $10.53 (19.86 hours) untuned. At this volume a per-token closed API is 10x cheaper per request than a dedicated card, so the case for self-hosting is data locality, offline operation and batching rather than the sticker price: this eval served one request at a time, and concurrent requests divide the GPU line while the token bill stays linear.
- The closed column prices the *tokens*, not a verified success rate: no closed model was run on these 90 Hinglish ledger tasks. If a closed model solved them at the base model's 3.3%, its bill per solved task would land near $55.63 instead of the per-request figure.

## Caveats a reader should apply

- The self-hosted side is charged at sequential, greedy decoding with model load inside the eval wall time. Prompt caching, a warm process, batching or ZeroGPU all lower it; none of that is assumed here.
- Self-hosting also costs engineering time, monitoring, cold starts and an eval harness. None of that is in the GPU line — it is excluded, not zero.
- These latencies are fp16 on one T4 (`--no-4bit`). A 4-bit server, which the demo Space uses, trades some speed for roughly a quarter of the memory; the earlier 4-bit round-1 measurement was 80.95 s/request untuned and 54.95 s tuned against 50.2 s/task here.
- This prices the merged model that was evaluated, not LoRA-adapter hosting (cheaper to serve, but it adds a PEFT dependency).
- Tiny Aya is not a frontier assistant: this is a measured cost and reliability trade-off on one closed task family, not a quality-parity claim.

## Assumptions and how to re-check

- GPU price is an on-demand T4 list price and is the only free parameter: `--gpu-hour` (here $0.53; Kaggle and HF ZeroGPU are $0 for this workload).
- Closed-model prices are published list prices checked on 2026-11-09 (openai.com/api/pricing; Nov-2026 LLM price comparison); they change, so treat them as a snapshot and edit `CLOSED` in `scripts/roi.py` to re-price.
- Token counts come from `runs/<run>/trajectories.jsonl` with `CohereLabs/tiny-aya-global`'s tokenizer, counting every assistant turn's prompt plus its generated message.

```bash
python scripts/roi.py --gpu-hour 0.53 --tasks-per-month 10000
```
