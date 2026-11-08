"""Rewrite a follow-up turn into a standalone query using recent history.

Every downstream node (intent, routing, retrieval, planner order-id regex) reads
``user_query``. Resolving "它 / 那个 / 那西藏呢" once, before intent classification,
fixes all of them without touching each node.
"""

from __future__ import annotations

import re

from pydantic import BaseModel

from app.core.exceptions import LLMOutputError, LLMProviderError
from app.memory.context import DEFAULT_CONTEXT_TURNS, render_history
from app.prompts.contextualize import CONTEXTUALIZE_SYSTEM, CONTEXTUALIZE_USER
from app.providers.base import LLMProvider
from app.providers.fake import OfflineLLMProvider
from app.providers.json_parser import parse_model
from app.schemas.chat import ChatMessage

_ORDER_ID = re.compile(r"ORD\d+", re.IGNORECASE)
_REFERENCE_CUES = (
    "它",
    "这个",
    "那个",
    "这单",
    "那单",
    "这笔",
    "那笔",
    "该订单",
    "上面",
    "刚才",
    "刚刚",
)
MAX_STANDALONE_CHARS = 300


class _Rewrite(BaseModel):
    standalone_query: str
    changed: bool = False


def rule_based_contextualize(query: str, messages: list[ChatMessage]) -> str:
    """Offline / failure fallback: carry the most recent order id into a referring turn."""
    if _ORDER_ID.search(query) or not any(cue in query for cue in _REFERENCE_CUES):
        return query
    for message in reversed(messages):
        found = _ORDER_ID.findall(message.content)
        if found:
            return f"{query}（订单 {found[-1].upper()}）"
    return query


class QueryContextualizer:
    def __init__(self, llm: LLMProvider, *, turns: int = DEFAULT_CONTEXT_TURNS) -> None:
        self._llm = llm
        self._turns = turns

    async def rewrite(self, query: str, messages: list[ChatMessage], summary: str = "") -> str:
        history = render_history(messages, summary, turns=self._turns)
        if not history:
            return query
        if isinstance(self._llm, OfflineLLMProvider):
            return rule_based_contextualize(query, messages)
        try:
            response = await self._llm.generate(
                [
                    {"role": "system", "content": CONTEXTUALIZE_SYSTEM},
                    {
                        "role": "user",
                        "content": CONTEXTUALIZE_USER.format(history=history, query=query),
                    },
                ],
                temperature=0.0,
                max_tokens=200,
                response_format={"type": "json_object"},
            )
            candidate = parse_model(response.content, _Rewrite).standalone_query.strip()
        except (LLMProviderError, LLMOutputError):
            return rule_based_contextualize(query, messages)
        if not candidate or len(candidate) > MAX_STANDALONE_CHARS:
            return rule_based_contextualize(query, messages)
        return candidate
