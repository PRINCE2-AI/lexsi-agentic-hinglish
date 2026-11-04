"""Create/refresh the Gradio Space with the right file layout.

A Space needs `app.py`, `README.md`, `requirements.txt` at the repo root and the
`hinglish_agent/` package next to them, which is not the layout of this repo.

    hf auth login               # browser flow, no token in the shell history
    python scripts/deploy_space.py --repo primce/hinglish-expense-agent

Then open the Space settings, pick a GPU (T4 small is enough), and wait for the build.
The app defaults to the shipped `sft3_adapter` (`ADAPTER_SUBFOLDER`); set `ADAPTER_ID` to an
empty value in the Space variables to demo the base model instead.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from huggingface_hub import HfApi, create_repo

ROOT = Path(__file__).resolve().parents[1]

FILES = {
    "space/app.py": "app.py",
    "space/requirements.txt": "requirements.txt",
}
FOLDERS = {"hinglish_agent": "hinglish_agent"}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", required=True, help="e.g. primce/hinglish-expense-agent")
    parser.add_argument("--private", action="store_true")
    args = parser.parse_args()

    token = os.environ.get("HF_TOKEN")
    if not token:
        from huggingface_hub.constants import HF_TOKEN_PATH

        if not Path(HF_TOKEN_PATH).exists():
            sys.exit(
                "Not logged in to Hugging Face. Run `hf auth login` (browser flow) or export "
                "HF_TOKEN in this shell; never paste a token into chat."
            )
    api = HfApi(token=token)
    create_repo(args.repo, repo_type="space", space_sdk="gradio", private=args.private, exist_ok=True)
    print(f"space ready: https://huggingface.co/spaces/{args.repo}")

    for source, target in FILES.items():
        api.upload_file(
            repo_id=args.repo,
            repo_type="space",
            path_or_fileobj=str(ROOT / source),
            path_in_repo=target,
        )
        print("uploaded", target)

    for source, target in FOLDERS.items():
        api.upload_folder(
            repo_id=args.repo,
            repo_type="space",
            folder_path=str(ROOT / source),
            path_in_repo=target,
            ignore_patterns=["__pycache__/*", "*.pyc"],
        )
        print("uploaded", target + "/")

    # space/README.md carries the Space metadata block (sdk, app_file), so it must be
    # uploaded as README.md last, after the code it points at.
    api.upload_file(
        repo_id=args.repo,
        repo_type="space",
        path_or_fileobj=str(ROOT / "space" / "README.md"),
        path_in_repo="README.md",
    )
    print("uploaded README.md")
    print()
    print("next: Space Settings -> Hardware -> T4 small (or ZeroGPU), then wait for the build.")


if __name__ == "__main__":
    main()
