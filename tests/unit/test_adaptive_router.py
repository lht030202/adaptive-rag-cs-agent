from app.agents.nodes.intent import rule_based_intent
from app.core.enums import IntentLabel, RetrievalStrategy
from app.retrieval.adaptive import rule_based_complexity
from app.schemas.agent import IntentResult


def test_greeting_goes_direct() -> None:
    intent = rule_based_intent("你好")
    assert intent.label == IntentLabel.GREETING
    routed = rule_based_complexity("你好", intent)
    assert routed.strategy == RetrievalStrategy.DIRECT


def test_order_goes_agent() -> None:
    intent = IntentResult(label=IntentLabel.ORDER_INQUIRY, confidence=0.9, prompt_version="t")
    routed = rule_based_complexity("查一下订单 ORD10001", intent)
    assert routed.strategy == RetrievalStrategy.AGENT


def test_policy_multi_hop() -> None:
    intent = IntentResult(label=IntentLabel.FAQ_POLICY, confidence=0.8, prompt_version="t")
    query = "定制马克杯过了7天但是质量问题还能退吗"
    routed = rule_based_complexity(query, intent)
    assert routed.strategy in {RetrievalStrategy.MULTI, RetrievalStrategy.SINGLE}


def test_general_return_policy_does_not_request_order_tools() -> None:
    intent = IntentResult(label=IntentLabel.RETURN_EXCHANGE, confidence=0.8, prompt_version="t")
    routed = rule_based_complexity("七天无理由退货的规则是什么？", intent)
    assert routed.strategy == RetrievalStrategy.SINGLE


def test_specific_order_return_uses_tools() -> None:
    intent = IntentResult(label=IntentLabel.RETURN_EXCHANGE, confidence=0.8, prompt_version="t")
    routed = rule_based_complexity("订单 ORD10002 还能退货吗？", intent)
    assert routed.strategy == RetrievalStrategy.AGENT


_LLM_SAYS_DIRECT = '{"complexity":"simple","strategy":"direct","confidence":0.9,"rationale":"llm"}'


async def test_llm_direct_on_policy_question_is_overridden() -> None:
    from app.providers.fake import FakeLLMProvider
    from app.retrieval.adaptive import AdaptiveRouter

    router = AdaptiveRouter(FakeLLMProvider({"电子发票": _LLM_SAYS_DIRECT}))
    intent = IntentResult(label=IntentLabel.FAQ_POLICY, confidence=0.9, prompt_version="t")
    routed = await router.route("电子发票多久能开", intent)
    assert routed.strategy == RetrievalStrategy.SINGLE
    assert routed.rationale.startswith("guard_direct_needs_evidence")


async def test_llm_direct_on_greeting_is_kept() -> None:
    from app.providers.fake import FakeLLMProvider
    from app.retrieval.adaptive import AdaptiveRouter

    router = AdaptiveRouter(FakeLLMProvider({"你好": _LLM_SAYS_DIRECT}))
    intent = IntentResult(label=IntentLabel.GREETING, confidence=0.9, prompt_version="t")
    routed = await router.route("你好", intent)
    assert routed.strategy == RetrievalStrategy.DIRECT


def test_intent_overlay_only_kept_offline_or_when_enabled() -> None:
    from app.agents.graph import use_intent_overlay
    from app.config.settings import Settings
    from app.providers.fake import FakeLLMProvider, OfflineLLMProvider

    off = Settings(_env_file=None, intent_overlay_enabled=False)
    on = Settings(_env_file=None, intent_overlay_enabled=True)
    assert use_intent_overlay(OfflineLLMProvider(), off) is True
    assert use_intent_overlay(FakeLLMProvider(), off) is False
    assert use_intent_overlay(FakeLLMProvider(), on) is True
