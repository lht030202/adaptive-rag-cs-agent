"""Multi-turn context: history rendering and standalone-query rewrite."""

from app.agents.nodes.contextualize import contextualize_node
from app.config.settings import Settings
from app.memory.context import render_history
from app.memory.contextualize import QueryContextualizer, rule_based_contextualize
from app.providers.fake import FakeLLMProvider, OfflineLLMProvider
from app.schemas.chat import ChatMessage

HISTORY = [
    ChatMessage(role="user", content="帮我查一下订单 ORD10003 现在什么状态"),
    ChatMessage(role="assistant", content="订单 ORD10003 已支付，仓库处理中。"),
]


def test_render_history_keeps_recent_turns_and_truncates() -> None:
    long_answer = ChatMessage(role="assistant", content="很长" * 300)
    text = render_history([*HISTORY, ChatMessage(role="user", content="q"), long_answer], turns=1)
    assert "ORD10003" not in text  # the older turn falls out of the window
    assert text.startswith("用户：q")
    assert text.endswith("…")


def test_render_history_is_empty_without_messages() -> None:
    assert render_history([], "") == ""


def test_rule_fallback_carries_order_id_into_reference() -> None:
    assert (
        rule_based_contextualize("那帮我把它取消掉", HISTORY) == "那帮我把它取消掉（订单 ORD10003）"
    )


def test_rule_fallback_leaves_complete_queries_alone() -> None:
    assert rule_based_contextualize("取消订单 ORD10001", HISTORY) == "取消订单 ORD10001"
    assert rule_based_contextualize("运费谁出", HISTORY) == "运费谁出"


async def test_llm_rewrite_is_used_when_valid() -> None:
    llm = FakeLLMProvider({"最新一句": '{"standalone_query":"取消订单 ORD10003","changed":true}'})
    assert (
        await QueryContextualizer(llm).rewrite("那帮我把它取消掉", HISTORY) == "取消订单 ORD10003"
    )


async def test_invalid_llm_output_falls_back_to_rules() -> None:
    llm = FakeLLMProvider()  # default canned reply is not a rewrite payload
    rewritten = await QueryContextualizer(llm).rewrite("那帮我把它取消掉", HISTORY)
    assert rewritten.endswith("（订单 ORD10003）")


async def test_first_turn_makes_no_llm_call() -> None:
    llm = FakeLLMProvider()
    assert await QueryContextualizer(llm).rewrite("你好", []) == "你好"
    assert llm.calls == []


async def test_node_respects_disable_switch() -> None:
    state = {"user_query": "那帮我把它取消掉", "messages": HISTORY, "summary": ""}
    contextualizer = QueryContextualizer(OfflineLLMProvider())
    off = await contextualize_node(
        state,  # type: ignore[arg-type]
        contextualizer=contextualizer,
        settings=Settings(_env_file=None, memory_context_enabled=False),
    )
    assert off == {"original_query": "那帮我把它取消掉"}
    on = await contextualize_node(
        state,  # type: ignore[arg-type]
        contextualizer=contextualizer,
        settings=Settings(_env_file=None, memory_context_enabled=True),
    )
    assert on["user_query"].endswith("（订单 ORD10003）")
    assert on["original_query"] == "那帮我把它取消掉"
