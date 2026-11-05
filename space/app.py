"""Gradio demo: Hinglish expense-splitting agent with a visible tool trace.

Local:
    python space/app.py

Hugging Face Space:
    MODEL_ID=CohereLabs/tiny-aya-global ADAPTER_ID=<you>/tiny-aya-hinglish-lora \
        ADAPTER_SUBFOLDER=sft4_adapter python app.py
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import gradio as gr

HERE = Path(__file__).resolve().parent
for candidate in (HERE, HERE.parent, Path.cwd()):
    if (candidate / "hinglish_agent").is_dir():
        sys.path.insert(0, str(candidate))
        break

from hinglish_agent.runner import run_episode  # noqa: E402

MODEL_ID = os.getenv("MODEL_ID", "CohereLabs/tiny-aya-global")
ADAPTER_ID = os.getenv("ADAPTER_ID", "primce/tiny-aya-hinglish-lora").strip() or None
ADAPTER_SUBFOLDER = os.getenv("ADAPTER_SUBFOLDER", "sft4_adapter").strip() or None
MAX_NEW_TOKENS = int(os.getenv("MAX_NEW_TOKENS", "256"))

_CACHE: dict = {}

try:  # ZeroGPU on Hugging Face Spaces
    if not os.getenv("SPACES_ZERO_GPU"):
        raise RuntimeError("not a ZeroGPU space")
    import spaces

    _gpu = spaces.GPU
except Exception:  # noqa: BLE001 - dedicated T4 spaces do not need the decorator
    _gpu = lambda fn: fn  # noqa: E731


def _load():
    if "model" in _CACHE:
        return _CACHE["model"], _CACHE["tokenizer"]

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    kwargs = {"device_map": "auto"}
    if torch.cuda.is_available():
        from transformers import BitsAndBytesConfig

        kwargs["torch_dtype"] = torch.float16
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4",
        )
    else:
        kwargs["torch_dtype"] = torch.float32

    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, **kwargs)
    if ADAPTER_ID:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, ADAPTER_ID, subfolder=ADAPTER_SUBFOLDER)
    model.eval()
    _CACHE["model"] = model
    _CACHE["tokenizer"] = tokenizer
    return model, tokenizer


def _make_call():
    import torch

    model, tokenizer = _load()

    def call(messages: list[dict]) -> str:
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=4096).to(model.device)
        with torch.no_grad():
            output = model.generate(
                **inputs,
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
            )
        generated = output[0][inputs["input_ids"].shape[1] :]
        return tokenizer.decode(generated, skip_special_tokens=True).strip()

    return call


def _trace_markdown(steps: list[dict]) -> str:
    lines = []
    for index, step in enumerate(steps, start=1):
        name = step.get("name") or "unparseable output"
        ok = bool(step.get("result", {}).get("ok"))
        lines.append(f"### {index}. `{name}` {'OK' if ok else 'ERROR'}")
        lines.append("arguments")
        lines.append("```json")
        lines.append(json.dumps(step.get("arguments", {}), ensure_ascii=False, indent=2)[:1500])
        lines.append("```")
        lines.append("result")
        lines.append("```json")
        lines.append(json.dumps(step.get("result", {}), ensure_ascii=False, indent=2)[:1500])
        lines.append("```")
    return "\n".join(lines) if lines else "No tool calls."


@_gpu
def run_agent(question: str):
    if not question or not question.strip():
        return "Sawal likho.", "No tool calls.", "{}"

    task = {"id": "demo", "task": question.strip()}
    episode = run_episode(_make_call(), task, max_steps=8)
    answer = episode["final_response"] or "(model did not call finish)"
    return answer, _trace_markdown(episode["steps"]), json.dumps(episode["ledger"], indent=2)


with gr.Blocks(title="Hinglish Expense Agent") as demo:
    gr.Markdown(
        "# Hinglish Expense-Splitting Agent\n"
        "Tiny Aya + AgentTune, Track 3. The agent calls tools, carries the ledger, "
        "and answers in Hinglish. Amounts in INR; USD = 83 INR, EUR = 90 INR.\n\n"
        "Shipped adapter: **96.67%** (29/30) on unseen hard tasks and **93.33%** (84/90) on a "
        "90-task holdout, scored on the final ledger with no LLM judge. It also ignores "
        "injected instructions in tool results: 0/24 ledger attacks and 0/24 attacker numbers "
        "in answers, against the base model's 13/24 and 8/24. Set `ADAPTER_ID` to an empty "
        "value to run the base model instead."
    )
    with gr.Row():
        with gr.Column(scale=2):
            question = gr.Textbox(
                label="Sawal",
                lines=3,
                value=(
                    "Rahul ne 1200 INR diye pizza ke liye. 4 log the: Rahul, Priya, Amit aur Neha. "
                    "Rahul ka final balance kitna hai?"
                ),
            )
            button = gr.Button("Run agent", variant="primary")
            answer = gr.Markdown()
        with gr.Column(scale=3):
            trace = gr.Markdown()
            ledger = gr.Code(language="json")

    button.click(run_agent, inputs=question, outputs=[answer, trace, ledger])
    question.submit(run_agent, inputs=question, outputs=[answer, trace, ledger])

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=int(os.getenv("PORT", "7860")))
