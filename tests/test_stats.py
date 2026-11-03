"""Stats checks. Run: python tests/test_stats.py

The reported gain has to survive a paired test, so Wilson bounds and the exact McNemar
tail are pinned to values computed by hand.
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.stats import compare, load_success, mcnemar_exact, wilson  # noqa: E402


def test_wilson_bounds() -> None:
    low, high = wilson(19, 30)
    assert 0.45 < low < 0.47, (low, high)
    assert 0.77 < high < 0.79, (low, high)

    low, high = wilson(30, 30)
    assert 0.88 < low < 0.90 and high > 0.999, (low, high)


def test_mcnemar_exact() -> None:
    assert abs(mcnemar_exact(10, 0) - 0.001953125) < 1e-9
    assert abs(mcnemar_exact(5, 5) - 1.0) < 1e-12
    assert mcnemar_exact(0, 0) == 1.0


def test_pairs_on_shared_ids_only() -> None:
    a = {"h-e-001": 1, "h-e-002": 0, "h-e-003": 0}
    b = {"h-e-001": 1, "h-e-002": 1, "h-e-004": 1}
    result = compare(a, b)
    assert result["n"] == 2, result
    assert result["a_success"] == 0.5
    assert result["b_success"] == 1.0
    assert result["b_only"] == 1 and result["a_only"] == 0
    assert result["dropped"] == ["h-e-003"]


def test_loader_reads_run_directory() -> None:
    row = {
        "metadata": {"task_id": "h-e-001", "scores": {"success": 1.0}},
        "final_response": "",
    }
    with tempfile.TemporaryDirectory() as tmp:
        run_dir = Path(tmp)
        (run_dir / "trajectories.jsonl").write_text(
            json.dumps(row) + "\n", encoding="utf-8"
        )
        assert load_success(run_dir) == {"h-e-001": 1}


def main() -> None:
    test_wilson_bounds()
    test_mcnemar_exact()
    test_pairs_on_shared_ids_only()
    test_loader_reads_run_directory()
    print("all stats tests passed")


if __name__ == "__main__":
    main()
