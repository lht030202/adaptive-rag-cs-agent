"""Repeat the e2e eval N rounds in one process and save a combined latency report.

Usage (project root, project venv):
    LLM_ENABLE_THINKING=false python scripts/e2e_latency.py thinking_off
    LLM_ENABLE_THINKING=true  python scripts/e2e_latency.py thinking_on --rounds 2

Environment variables override .env, so the same code can be measured with the
switch on and off. All project code is imported once at start-up, so editing the
source while this runs cannot leak into the measurement.
"""

import argparse
import asyncio
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from eval.metrics import percentile  # noqa: E402
from eval.runner import REPORT_DIR, eval_e2e  # noqa: E402


async def main(label: str, rounds: int) -> None:
    runs = []
    for index in range(rounds):
        result = await eval_e2e()
        runs.append(result)
        lat = result["latency_ms"]
        print(
            f"round {index + 1}: P50={lat['p50'] / 1000:.1f}s P95={lat['p95'] / 1000:.1f}s "
            f"keyword_hit={result['keyword_hit']} refuse_acc={result['refuse_accuracy']} "
            f"tool_path={result['tool_path_hit']}",
            flush=True,
        )
    latencies = [case["latency_ms"] for run in runs for case in run["cases"]]
    combined = {
        "n": len(latencies),
        "p50_ms": round(percentile(latencies, 50), 1),
        "p95_ms": round(percentile(latencies, 95), 1),
        "mean_ms": round(sum(latencies) / len(latencies), 1),
        "keyword_hit": [run["keyword_hit"] for run in runs],
        "refuse_accuracy": [run["refuse_accuracy"] for run in runs],
        "tool_path_hit": [run["tool_path_hit"] for run in runs],
    }
    print("combined:", json.dumps(combined, ensure_ascii=False))
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    out = REPORT_DIR / f"e2e_{label}_{datetime.now(UTC):%Y%m%dT%H%M%SZ}.json"
    out.write_text(
        json.dumps(
            {"label": label, "rounds": runs, "combined": combined}, ensure_ascii=False, indent=2
        ),
        encoding="utf-8",
    )
    print("wrote", out)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("label")
    parser.add_argument("--rounds", type=int, default=2)
    args = parser.parse_args()
    asyncio.run(main(args.label, args.rounds))
