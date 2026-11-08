"""Render recent conversation turns into a compact, budgeted prompt block."""

from __future__ import annotations

from app.schemas.chat import ChatMessage

DEFAULT_CONTEXT_TURNS = 3
MAX_CHARS_PER_MESSAGE = 200
MAX_SUMMARY_CHARS = 400
_ROLE_LABELS = {"user": "用户", "assistant": "客服"}


def recent_dialogue(
    messages: list[ChatMessage], *, turns: int = DEFAULT_CONTEXT_TURNS
) -> list[ChatMessage]:
    """Last ``turns`` user/assistant exchanges; system and tool messages are dropped."""
    if turns <= 0:
        return []
    dialogue = [message for message in messages if message.role in _ROLE_LABELS]
    return dialogue[-turns * 2 :]


def has_prior_user_turn(messages: list[ChatMessage]) -> bool:
    return any(message.role == "user" for message in messages)


def render_history(
    messages: list[ChatMessage],
    summary: str = "",
    *,
    turns: int = DEFAULT_CONTEXT_TURNS,
    max_chars: int = MAX_CHARS_PER_MESSAGE,
) -> str:
    """Return '' when there is nothing to show, so callers can skip the extra work.

    Each message is whitespace-collapsed and cut at ``max_chars`` so a long tool-heavy
    answer cannot crowd the prompt; older turns survive only through ``summary``.
    """
    lines: list[str] = []
    if summary:
        lines.append(f"更早的对话摘要：{summary[:MAX_SUMMARY_CHARS]}")
    for message in recent_dialogue(messages, turns=turns):
        text = " ".join(message.content.split())
        if len(text) > max_chars:
            text = text[:max_chars] + "…"
        lines.append(f"{_ROLE_LABELS[message.role]}：{text}")
    return "\n".join(lines)
