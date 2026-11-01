# ROI comparison template (optional but preferred by judges)

Use current prices on the day you submit. Label every assumption. Do not copy these placeholders as facts.

## Scenario

- Monthly agent requests: [10,000]
- Average input tokens per request: [800]
- Average output tokens per request: [200]
- Average tool-call turns per request: [3]

## Closed-source API

```
monthly_cost = requests * (input_tokens * input_price + output_tokens * output_price) / 1_000_000
```

Fill current per-million prices for the API you compare against: [input price], [output price].
Result: [$X/month].

## Self-hosted Tiny Aya

```
gpu_hours = requests * seconds_per_request * turns / 3600
monthly_cost = gpu_hours * gpu_hour_price
```

- GPU: [T4 / L4 / A10G], [$ per hour]
- Throughput measured on your machine: [tokens/sec]
- Seconds per request measured: [X]
- Result: [$Y/month]

## Break-even

```
break_even_requests = fixed_monthly_cost / (api_cost_per_request - self_host_cost_per_request)
```

Result: [N requests/month].

## Honest caveats

- Tiny Aya is less capable than frontier APIs; this is a cost/reliability trade-off, not a quality parity claim.
- Self-hosting includes engineering time, monitoring and cold-start latency; add them or say you excluded them.
- LoRA adapter hosting is cheaper than full-model hosting; state which one you priced.
