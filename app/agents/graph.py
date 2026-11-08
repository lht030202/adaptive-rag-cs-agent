"""LangGraph state machine that wires the agent nodes together."""

from __future__ import annotations

from functools import partial

from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from app.agents.nodes.contextualize import contextualize_node
from app.agents.nodes.executor import executor_node
from app.agents.nodes.faithfulness import faithfulness_node
from app.agents.nodes.generate import generate_node, refuse_node
from app.agents.nodes.guardrail import guardrail_node
from app.agents.nodes.intent import IntentClassifier
from app.agents.nodes.memory import memory_node
from app.agents.nodes.planner import Planner, planner_node
from app.agents.nodes.retrieval import retrieval_node
from app.agents.nodes.router import intent_node, router_node
from app.agents.state import AgentState
from app.config.settings import Settings
from app.core.enums import RetrievalStrategy
from app.guardrails.hallucination import FaithfulnessChecker
from app.memory.contextualize import QueryContextualizer
from app.memory.summarizer import ConversationSummarizer
from app.providers.base import LLMProvider
from app.providers.fake import OfflineLLMProvider
from app.retrieval.adaptive import AdaptiveRouter
from app.retrieval.hybrid import HybridRetriever
from app.retrieval.query_rewrite import QueryRewriter
from app.retrieval.reranker import DocumentReranker
from app.retrieval.self_rag import SelfRAGLoop
from app.tools.registry import ToolRegistry


def use_intent_overlay(llm: LLMProvider, settings: Settings) -> bool:
    """Keep the lexical overlay only where it helps.

    On the held-out intent set the overlay lowered LLM accuracy (96.9% -> 88.8%),
    because its phrases were written from the original test set. Without a model
    (offline mode) the overlay is still the best rule system available.
    """
    return settings.intent_overlay_enabled or isinstance(llm, OfflineLLMProvider)


def _route_after_complexity(state: AgentState) -> str:
    strategy = state.get("strategy") or RetrievalStrategy.SINGLE
    mapping = {
        RetrievalStrategy.DIRECT: "generate_answer",
        RetrievalStrategy.REFUSE: "refuse_answer",
        RetrievalStrategy.SINGLE: "retrieve_docs",
        RetrievalStrategy.MULTI: "retrieve_docs",
        RetrievalStrategy.AGENT: "plan_tasks",
    }
    return mapping[strategy]


def build_graph(
    *,
    llm: LLMProvider,
    settings: Settings,
    retriever: HybridRetriever,
    reranker: DocumentReranker,
    registry: ToolRegistry,
) -> CompiledStateGraph:
    classifier = IntentClassifier(llm, apply_overlay=use_intent_overlay(llm, settings))
    adaptive = AdaptiveRouter(llm)
    rewriter = QueryRewriter(llm)
    self_rag = SelfRAGLoop(retriever, reranker, rewriter, llm, settings)
    planner = Planner(llm, registry)
    checker = FaithfulnessChecker(llm)
    summarizer = ConversationSummarizer(llm)
    contextualizer = QueryContextualizer(llm, turns=settings.memory_context_turns)

    graph = StateGraph(AgentState)
    graph.add_node("apply_guardrail", guardrail_node)
    graph.add_node(
        "contextualize_query",
        partial(contextualize_node, contextualizer=contextualizer, settings=settings),
    )
    graph.add_node("classify_intent", partial(intent_node, classifier=classifier))
    graph.add_node("route_complexity", partial(router_node, router=adaptive))
    graph.add_node(
        "retrieve_docs",
        partial(
            retrieval_node,
            retriever=retriever,
            reranker=reranker,
            rewriter=rewriter,
            self_rag=self_rag,
            settings=settings,
        ),
    )
    graph.add_node("plan_tasks", partial(planner_node, planner=planner))
    graph.add_node("execute_tools", partial(executor_node, registry=registry, settings=settings))
    graph.add_node("generate_answer", partial(generate_node, llm=llm, settings=settings))
    graph.add_node("refuse_answer", refuse_node)
    graph.add_node("check_faithfulness", partial(faithfulness_node, checker=checker, llm=llm))
    graph.add_node("compact_memory", partial(memory_node, settings=settings, summarizer=summarizer))

    def _after_guardrail(state: AgentState) -> str:
        return "refuse_answer" if state.get("refused") else "contextualize_query"

    graph.add_edge(START, "apply_guardrail")
    graph.add_conditional_edges(
        "apply_guardrail",
        _after_guardrail,
        {"refuse_answer": "refuse_answer", "contextualize_query": "contextualize_query"},
    )
    # Guardrail checks the raw text first; the rewrite then resolves references so
    # intent, routing, retrieval and the planner all see a standalone question.
    graph.add_edge("contextualize_query", "classify_intent")
    graph.add_edge("classify_intent", "route_complexity")
    graph.add_conditional_edges(
        "route_complexity",
        _route_after_complexity,
        {
            "generate_answer": "generate_answer",
            "refuse_answer": "refuse_answer",
            "retrieve_docs": "retrieve_docs",
            "plan_tasks": "plan_tasks",
        },
    )
    graph.add_edge("retrieve_docs", "generate_answer")
    graph.add_edge("plan_tasks", "execute_tools")
    graph.add_edge("execute_tools", "generate_answer")
    graph.add_edge("generate_answer", "check_faithfulness")
    graph.add_edge("refuse_answer", "compact_memory")
    graph.add_edge("check_faithfulness", "compact_memory")
    graph.add_edge("compact_memory", END)
    return graph.compile()
