---
language:
  - hi
  - en
base_model: CohereLabs/tiny-aya-global
license: cc-by-nc-4.0
library_name: transformers
tags:
  - agent
  - tool-calling
  - hinglish
  - lora
  - merged
datasets:
  - primce/lexsi-runs
---

# tiny-aya-hinglish-merged

`CohereLabs/tiny-aya-global` with the shipped LoRA adapter
[`primce/tiny-aya-hinglish-lora`](https://huggingface.co/primce/tiny-aya-hinglish-lora)
merged into the base weights. Same network, no PEFT:

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained("primce/tiny-aya-hinglish-merged", dtype="float16")
tokenizer = AutoTokenizer.from_pretrained("primce/tiny-aya-hinglish-merged")
```

Everything that describes behaviour, data, safety numbers and limitations lives in the adapter
card and the write-up; this repository exists so the same weights load without installing `peft`.

- Adapter card: https://huggingface.co/primce/tiny-aya-hinglish-lora
- Write-up: https://huggingface.co/spaces/primce/hinglish-track3-article
- Run logs behind every number: https://huggingface.co/datasets/primce/lexsi-runs
- Demo: https://huggingface.co/spaces/primce/hinglish-expense-agent

This is not a second model and not a second result. It is the published adapter with
`merge_and_unload()` applied, and the run log `merged_parity` in `primce/lexsi-runs` checks that the
merged weights reproduce the adapter's per-task verdicts on the same three tasks.

Licence: the adapter and these merged weights are **CC BY-NC 4.0**, inherited from
`CohereLabs/tiny-aya-global`. The evaluation code in the GitHub repository is MIT.
