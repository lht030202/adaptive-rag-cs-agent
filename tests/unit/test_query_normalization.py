"""Queries must be cleaned the same way as the indexed chunks.

Chunks go through normalize_text at ingest (NFKC turns full-width "５９" into "59").
If the query skips that step, full-width input stops matching the index.
"""

from app.core.enums import DocumentCategory
from app.retrieval.bm25 import BM25Retriever
from app.retrieval.chunking import chunk_markdown
from app.retrieval.reranker import DocumentReranker
from app.retrieval.vector import VectorRetriever

_POLICY = (
    "## 包邮\n\n满59元包邮，偏远地区除外。\n\n"
    "## 退货\n\n定制商品不适用7天无理由。\n\n"
    "## 发票\n\n电子发票在确认收货后开具。\n\n"
    "## 积分\n\n积分可以抵扣部分货款。\n\n"
    "## 保修\n\n手机享受一年保修。"
)


class _RecordingEmbedder:
    def __init__(self) -> None:
        self.seen: list[str] = []

    async def embed(self, texts: list[str]) -> list[list[float]]:
        self.seen.extend(texts)
        return [[1.0, 0.0] for _ in texts]


class _EmptyStore:
    def search(self, query_vector: list[float], *, top_k: int) -> list:
        return []


class _RecordingReranker:
    def __init__(self) -> None:
        self.queries: list[str] = []

    async def rerank(
        self, query: str, documents: list[str], *, top_n: int
    ) -> list[tuple[int, float]]:
        self.queries.append(query)
        return [(index, 1.0) for index in range(top_n)]


def test_bm25_full_width_query_scores_like_half_width() -> None:
    chunks = chunk_markdown(
        _POLICY,
        document_id="policy",
        title="policy",
        category=DocumentCategory.SHIPPING,
        source="x",
    )
    retriever = BM25Retriever(chunks)
    half = retriever.search("满59包邮吗", top_k=1)[0]
    full = retriever.search("满５９包邮吗", top_k=1)[0]
    assert full.chunk_id == half.chunk_id
    assert full.score == half.score


async def test_vector_search_embeds_the_cleaned_query() -> None:
    embedder = _RecordingEmbedder()
    retriever = VectorRetriever(_EmptyStore(), embedder)  # type: ignore[arg-type]
    await retriever.search("满５９包邮吗", top_k=3)
    assert embedder.seen == ["满59包邮吗"]


async def test_reranker_scores_the_cleaned_query() -> None:
    chunks = chunk_markdown(
        _POLICY,
        document_id="policy",
        title="policy",
        category=DocumentCategory.SHIPPING,
        source="x",
    )
    provider = _RecordingReranker()
    await DocumentReranker(provider).rerank("满５９包邮吗", chunks[:2], top_n=2)
    assert provider.queries == ["满59包邮吗"]
