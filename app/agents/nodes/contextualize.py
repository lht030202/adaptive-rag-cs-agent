"""Multi-turn node: resolve references in the latest turn before intent and routing."""

from __future__ import annotations

from app.agents.state import AgentState
from app.config.settings import Settings
from app.memory.context import has_prior_user_turn
from app.memory.contextualize import QueryContextualizer


async def contextualize_node(
    state: AgentState,
    *,
    contextualizer: QueryContextualizer,
    settings: Settings,
) -> dict:
    """Replace ``user_query`` with a standalone question; keep the raw text separately.

    Skipped (no LLM call) on the first turn, on a confirmation turn, and when
    MEMORY_CONTEXT_ENABLED=false, which is how the multi-turn eval measures "before".
    """
    query = state["user_query"]
    messages = state.get("messages") or []
    if (
        not settings.memory_context_enabled
        or state.get("confirmed_action") is not None
        or not has_prior_user_turn(messages)
    ):
        return {"original_query": query}
    standalone = await contextualizer.rewrite(query, messages, state.get("summary") or "")
    if standalone == query:
        return {"original_query": query}
    return {"user_query": standalone, "original_query": query}
