"""Judge the SAME generated answers with several judge models and compare them.

Generating once and judging the identical answers isolates judge behaviour from
generation randomness, so the gap between judges (e.g. self-preference of the
generator judging itself) and their agreement (Cohen's kappa) are measured cleanly.

Usage (project root, project venv):
    python scripts/judge_compare.py
    python scripts/judge_compare.py --judges Qwen/Qwen3.5-122B-A10B Qwen/Qwen3.8-27B
"""

import argparse
import asyncio
import json
import sys
from datetime import UTC, datetime
from itertools import combinations
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.config.settings import get_settings  # noqa: E402
from app.guardrails.hallucination import FaithfulnessChecker  # noqa: E402
from app.providers.deepseek import DeepSeekProvider  # noqa: E402
from app.providers.embeddings import build_embedding_provider  # noqa: E402
from app.retrieval.hybrid import HybridRetriever  # noqa: E402
from app.retrieval.ingest import ingest_to_memory  # noqa: E402
from app.retrieval.vector import VectorRetriever  # noqa: E402
from eval.runner import RAG_PATH, REPORT_DIR, _generate_answer, load_jsonl  # noqa: E402

DEFAULT_JUDGES = [
    "Qwen/Qwen3.5-122B-A10B",
    "Qwen/Qwen3.8-27B",
    "deepseek-ai/DeepSeek-V4.1-Flash",
]
CONDITIONS = ("naive", "rag")


def cohen_kappa(a: list[bool], b: list[bool]) -> float:
    n = len(a)
    if n == 0:
        return 0.0
    observed = sum(x == y for x, y in zip(a, b, strict=True)) / n
    pa, pb = sum(a) / n, sum(b) / n
    expected = pa * pb + (1 - pa) * (1 - pb)
    return (observed - expected) / (1 - expected) if expected < 1 else 1.0


async def main(judge_models: list[str], concurrency: int) -> None:
    settings = get_settings()
    generator = DeepSeekProvider(settings)
    judges = {
        model: FaithfulnessChecker(
            generator
            if model == settings.deepseek_model
            else DeepSeekProvider(settings.model_copy(update={"deepseek_model": model}))
        )
        for model in judge_models
    }
    embedder = build_embedding_provider(settings)
    bm25, store, _ = await ingest_to_memory("knowledge", embedder)
    retriever = HybridRetriever(bm25, VectorRetriever(store, embedder), rrf_k=60)
    rows = load_jsonl(RAG_PATH)
    semaphore = asyncio.Semaphore(concurrency)
    items: list[dict] = []

    async def judge_all(query: str, answer: str, chunks: list) -> dict[str, bool | None]:
        verdicts = await asyncio.gather(
            *[
                checker.check(query=query, answer=answer, chunks=chunks, tool_results_text="")
                for checker in judges.values()
            ]
        )
        return {
            model: (None if v.rationale == "checker_unavailable" else bool(v.supported))
            for model, v in zip(judges, verdicts, strict=True)
        }

    async def one(row: dict) -> None:
        async with semaphore:
            chunks = await retriever.retrieve(row["query"], top_k=5)
            evidence = (
                "\n\n".join(f"[{c.chunk_id}] {c.content}" for c in chunks[:6]) or "(无检索结果)"
            )
            naive, rag = await asyncio.gather(
                _generate_answer(generator, row["query"], "(无检索结果)"),
                _generate_answer(generator, row["query"], evidence),
            )
            answers = {"naive": naive, "rag": rag}
            verdicts = {
                cond: await judge_all(row["query"], answers[cond], chunks) for cond in CONDITIONS
            }
            items.append(
                {
                    "id": row["id"],
                    "query": row["query"],
                    "answers": answers,
                    "verdicts": verdicts,
                    "evidence": evidence[:1500],
                }
            )

    await asyncio.gather(*[one(row) for row in rows])
    items.sort(key=lambda item: item["id"])

    summary: dict = {"n": len(items), "generator": settings.deepseek_model, "per_judge": {}}
    for model in judge_models:
        stats = {}
        for cond in CONDITIONS:
            values = [item["verdicts"][cond][model] for item in items]
            judged = [v for v in values if v is not None]
            stats[cond] = {
                "faithfulness_of_judged": round(sum(judged) / len(judged), 4) if judged else None,
                "judged": len(judged),
                "judge_errors": values.count(None),
            }
        summary["per_judge"][model] = stats
    agreement = {}
    for a, b in combinations(judge_models, 2):
        pairs = [
            (item["verdicts"][cond][a], item["verdicts"][cond][b])
            for item in items
            for cond in CONDITIONS
            if item["verdicts"][cond][a] is not None and item["verdicts"][cond][b] is not None
        ]
        xs = [bool(x) for x, _ in pairs]
        ys = [bool(y) for _, y in pairs]
        agreement[f"{a} vs {b}"] = {
            "n": len(pairs),
            "agreement": round(sum(x == y for x, y in pairs) / len(pairs), 4) if pairs else None,
            "cohen_kappa": round(cohen_kappa(xs, ys), 4),
        }
    summary["agreement"] = agreement
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    out = REPORT_DIR / f"judge_compare_{datetime.now(UTC):%Y%m%dT%H%M%SZ}.json"
    out.write_text(
        json.dumps({"summary": summary, "items": items}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print("wrote", out)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--judges", nargs="+", default=DEFAULT_JUDGES)
    parser.add_argument("--concurrency", type=int, default=5)
    args = parser.parse_args()
    asyncio.run(main(args.judges, args.concurrency))
