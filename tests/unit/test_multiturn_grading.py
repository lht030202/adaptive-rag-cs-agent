"""Grading rules for the multi-turn eval (eval/runner.py: multiturn_pass)."""

from eval.runner import MULTITURN_PATH, load_jsonl, multiturn_pass


def test_pending_action_must_match_tool_and_order() -> None:
    expect = {"pending_tool": "cancel_order", "order_id": "ORD10003"}
    assert multiturn_pass(expect, "确认后将取消订单 ORD10003。", "cancel_order", "ORD10003")
    assert not multiturn_pass(expect, "请提供订单号", None, None)
    assert not multiturn_pass(expect, "确认后将取消订单 ORD10001。", "cancel_order", "ORD10001")


def test_keyword_rules_all_and_any() -> None:
    expect = {"all": ["西藏"], "any": ["偏远", "运费"]}
    assert multiturn_pass(expect, "西藏属于偏远地区，需要加收运费", None, None)
    assert not multiturn_pass(expect, "西藏的问题请转人工", None, None)
    assert not multiturn_pass(expect, "偏远地区需要加收运费", None, None)


def test_dataset_rows_are_well_formed() -> None:
    rows = load_jsonl(MULTITURN_PATH)
    assert len(rows) >= 20
    for row in rows:
        assert len(row["turns"]) >= 2
        assert row["expect"]


async def test_generation_eval_prompt_formats_with_all_fields() -> None:
    """Guards the eval's copy of the answer prompt against new template fields."""
    from app.providers.fake import FakeLLMProvider
    from eval.runner import _generate_answer

    llm = FakeLLMProvider({"用户问题": '{"answer":"ok","citation_chunk_ids":[]}'})
    assert await _generate_answer(llm, "运费谁出", "(无检索结果)") == "ok"
