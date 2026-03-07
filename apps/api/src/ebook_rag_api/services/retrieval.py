import re
from dataclasses import dataclass
from math import log

from sqlalchemy import desc, func, select
from sqlalchemy.orm import Session, joinedload

from ebook_rag_api.core.config import get_settings
from ebook_rag_api.db.vector import is_postgresql_dialect
from ebook_rag_api.models import Document, DocumentChunk
from ebook_rag_api.services.embeddings import get_embedding_provider
from ebook_rag_api.services.reranking import TokenOverlapReranker, get_reranker
from ebook_rag_api.services.text import (
    contains_normalized_phrase,
    extract_anchor_terms,
    extract_constraint_terms,
    metadata_noise_score,
    normalize_query_text,
    normalized_token_sequence,
    query_run_bonus,
    tokenize_terms,
)

YEAR_RE = re.compile(r"\b\d{4}\b")
MONTH_RE = re.compile(
    r"\b(?:jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|"
    r"jul(?:y)?|aug(?:ust)?|sep(?:tember)?|oct(?:ober)?|nov(?:ember)?|"
    r"dec(?:ember)?)\b",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class ChunkSearchMatch:
    chunk: DocumentChunk
    dense_score: float
    lexical_score: float
    hybrid_score: float
    rerank_score: float
    score: float


@dataclass(frozen=True)
class HybridCandidate:
    chunk: DocumentChunk
    dense_score: float
    lexical_score: float
    hybrid_score: float


def normalize_query(text: str) -> str:
    return normalize_query_text(text)


def cosine_similarity(left: list[float], right: list[float]) -> float:
    import math

    if not left or not right or len(left) != len(right):
        return 0.0

    dot_product = sum(a * b for a, b in zip(left, right, strict=False))
    left_norm = math.sqrt(sum(value * value for value in left))
    right_norm = math.sqrt(sum(value * value for value in right))
    if left_norm == 0 or right_norm == 0:
        return 0.0
    return dot_product / (left_norm * right_norm)


def search_chunks(
    session: Session, query: str, top_k: int, document_id: str | None = None
) -> tuple[str, list[ChunkSearchMatch]]:
    normalized_query = normalize_query(query)
    if not normalized_query:
        return "", []

    settings = get_settings()
    provider = get_embedding_provider()
    query_embedding = provider.embed_texts([normalized_query])[0]
    candidate_limit = max(top_k, top_k * settings.rerank_candidate_multiplier)

    dense_matches = search_dense_candidates(
        session=session,
        query_embedding=query_embedding,
        candidate_limit=candidate_limit,
        document_id=document_id,
    )
    lexical_matches = (
        search_lexical_candidates(
            session=session,
            query=normalized_query,
            candidate_limit=candidate_limit,
            document_id=document_id,
        )
        if settings.retrieval_enable_lexical
        else []
    )

    fused_candidates = fuse_candidates(
        query=normalized_query,
        dense_matches=dense_matches,
        lexical_matches=lexical_matches,
        candidate_limit=candidate_limit,
    )
    return normalized_query, rerank_matches(
        query=normalized_query,
        fused_candidates=fused_candidates,
        top_k=top_k,
    )


def search_dense_candidates(
    session: Session,
    query_embedding: list[float],
    candidate_limit: int,
    document_id: str | None = None,
) -> list[tuple[DocumentChunk, float]]:
    bind = session.get_bind()
    if bind is not None and is_postgresql_dialect(bind.dialect.name):
        score_expression = (1 - DocumentChunk.embedding_vector.cosine_distance(query_embedding)).label(
            "score"
        )
        statement = (
            select(DocumentChunk, score_expression)
            .join(Document)
            .options(joinedload(DocumentChunk.document))
            .where(Document.status == "ready")
            .where(DocumentChunk.embedding_vector.is_not(None))
            .where(DocumentChunk.embedding_dimensions == len(query_embedding))
        )
        if document_id is not None:
            statement = statement.where(DocumentChunk.document_id == document_id)

        statement = statement.order_by(
            desc(score_expression),
            DocumentChunk.page_start.asc(),
            DocumentChunk.chunk_index.asc(),
        ).limit(candidate_limit)
        rows = session.execute(statement).all()
        return [(row[0], float(row[1])) for row in rows]

    candidates = load_ready_chunks(session=session, document_id=document_id)
    scored_matches: list[tuple[DocumentChunk, float]] = []
    for chunk in candidates:
        if (
            chunk.embedding_vector is None
            or chunk.embedding_dimensions != len(query_embedding)
        ):
            continue
        score = cosine_similarity(query_embedding, chunk.embedding_vector)
        scored_matches.append((chunk, score))

    scored_matches.sort(key=lambda item: (-item[1], item[0].page_start, item[0].chunk_index))
    return scored_matches[:candidate_limit]


def search_lexical_candidates(
    session: Session,
    query: str,
    candidate_limit: int,
    document_id: str | None = None,
) -> list[tuple[DocumentChunk, float]]:
    if not query:
        return []

    bind = session.get_bind()
    if bind is not None and is_postgresql_dialect(bind.dialect.name):
        query_text = func.concat(func.coalesce(DocumentChunk.heading, ""), " ", DocumentChunk.text)
        ts_vector = func.to_tsvector("english", query_text)
        ts_query = func.plainto_tsquery("english", query)
        score_expression = func.ts_rank_cd(ts_vector, ts_query).label("lexical_score")
        statement = (
            select(DocumentChunk, score_expression)
            .join(Document)
            .options(joinedload(DocumentChunk.document))
            .where(Document.status == "ready")
            .where(score_expression > 0)
        )
        if document_id is not None:
            statement = statement.where(DocumentChunk.document_id == document_id)

        statement = statement.order_by(
            desc(score_expression),
            DocumentChunk.page_start.asc(),
            DocumentChunk.chunk_index.asc(),
        ).limit(candidate_limit)
        rows = session.execute(statement).all()
        return [(row[0], float(row[1])) for row in rows]

    scored_matches: list[tuple[DocumentChunk, float]] = []
    for chunk in load_ready_chunks(session=session, document_id=document_id):
        score = lexical_overlap_score(query=query, text=chunk.text, heading=chunk.heading)
        if score <= 0:
            continue
        scored_matches.append((chunk, score))

    scored_matches.sort(key=lambda item: (-item[1], item[0].page_start, item[0].chunk_index))
    return scored_matches[:candidate_limit]


def load_ready_chunks(
    session: Session,
    document_id: str | None = None,
) -> list[DocumentChunk]:
    statement = (
        select(DocumentChunk)
        .join(Document)
        .options(joinedload(DocumentChunk.document))
        .where(Document.status == "ready")
    )
    if document_id is not None:
        statement = statement.where(DocumentChunk.document_id == document_id)
    return list(session.scalars(statement))


def lexical_overlap_score(query: str, text: str, heading: str | None = None) -> float:
    query_terms = tokenize_for_search(query)
    if not query_terms:
        return 0.0

    searchable_text = f"{heading or ''} {text}".strip()
    passage_terms = tokenize_for_search(searchable_text)
    if not passage_terms:
        return 0.0

    overlap = query_terms & passage_terms
    if not overlap:
        return 0.0

    anchor_terms = extract_anchor_terms(query)
    anchor_overlap = anchor_terms & passage_terms
    constraint_terms = extract_constraint_terms(query)
    constraint_overlap = constraint_terms & passage_terms
    coverage = len(overlap) / len(query_terms)
    precision = len(overlap) / len(passage_terms)
    anchor_coverage = len(anchor_overlap) / len(anchor_terms) if anchor_terms else 0.0
    anchor_precision = len(anchor_overlap) / len(passage_terms) if anchor_overlap else 0.0
    constraint_coverage = (
        len(constraint_overlap) / len(constraint_terms) if constraint_terms else 0.0
    )
    constraint_precision = (
        len(constraint_overlap) / len(passage_terms) if constraint_overlap else 0.0
    )
    phrase_bonus = 0.15 if contains_query_phrase(query, searchable_text) else 0.0
    query_run_match_bonus = query_run_bonus(query, searchable_text, max_bonus=0.2)
    heading_bonus = 0.1 if heading and (query_terms & tokenize_for_search(heading)) else 0.0
    anchor_bonus = (
        anchor_coverage * 0.35
        + anchor_precision * 0.1
        + _ordered_anchor_pair_bonus(query, searchable_text)
    )
    constraint_bonus = constraint_coverage * 0.35 + constraint_precision * 0.08
    metadata_penalty = metadata_noise_score(searchable_text) * 0.18
    return max(
        0.0,
        coverage * 0.32
        + precision * 0.08
        + phrase_bonus
        + query_run_match_bonus
        + heading_bonus
        + anchor_bonus
        + constraint_bonus
        - metadata_penalty,
    )


def tokenize_for_search(text: str) -> set[str]:
    return tokenize_terms(text)


def contains_query_phrase(query: str, text: str) -> bool:
    return contains_normalized_phrase(query, text)


def fuse_candidates(
    query: str,
    dense_matches: list[tuple[DocumentChunk, float]],
    lexical_matches: list[tuple[DocumentChunk, float]],
    candidate_limit: int,
) -> list[HybridCandidate]:
    settings = get_settings()
    dense_ranks = {chunk.id: index for index, (chunk, _) in enumerate(dense_matches, start=1)}
    lexical_ranks = {chunk.id: index for index, (chunk, _) in enumerate(lexical_matches, start=1)}
    dense_scores = {chunk.id: score for chunk, score in dense_matches}
    lexical_scores = {chunk.id: score for chunk, score in lexical_matches}
    chunks_by_id = {chunk.id: chunk for chunk, _ in dense_matches}
    chunks_by_id.update({chunk.id: chunk for chunk, _ in lexical_matches})
    query_terms = tokenize_for_search(query)
    candidate_term_sets = {
        chunk_id: tokenize_for_search(f"{chunk.heading or ''} {chunk.text}".strip())
        for chunk_id, chunk in chunks_by_id.items()
    }
    term_document_frequency = {
        term: sum(1 for terms in candidate_term_sets.values() if term in terms)
        for term in query_terms
    }

    raw_hybrid_scores: dict[str, float] = {}
    for chunk_id in chunks_by_id:
        dense_component = reciprocal_rank_score(
            rank=dense_ranks.get(chunk_id),
            weight=settings.retrieval_dense_weight,
            rrf_k=settings.retrieval_rrf_k,
        )
        lexical_component = reciprocal_rank_score(
            rank=lexical_ranks.get(chunk_id),
            weight=settings.retrieval_lexical_weight,
            rrf_k=settings.retrieval_rrf_k,
        )
        specificity_bonus = min(
            1.0,
            lexical_overlap_score(
                query=query,
                text=chunks_by_id[chunk_id].text,
                heading=chunks_by_id[chunk_id].heading,
            ),
        )
        focus_bonus = _focused_query_match_bonus(
            query=query,
            text=f"{chunks_by_id[chunk_id].heading or ''} {chunks_by_id[chunk_id].text}".strip(),
        )
        rarity_bonus = _query_term_rarity_bonus(
            query_terms=query_terms,
            passage_terms=candidate_term_sets.get(chunk_id, set()),
            term_document_frequency=term_document_frequency,
            candidate_count=max(len(candidate_term_sets), 1),
        )
        raw_hybrid_scores[chunk_id] = max(
            0.0,
            dense_component
            + lexical_component
            + specificity_bonus * 0.3
            + focus_bonus * 0.28
            + rarity_bonus * 0.25
            + query_run_bonus(
                query,
                f"{chunks_by_id[chunk_id].heading or ''} {chunks_by_id[chunk_id].text}".strip(),
                max_bonus=0.2,
            )
            - metadata_noise_score(chunks_by_id[chunk_id].text) * 0.3,
        )

    max_hybrid_score = max(raw_hybrid_scores.values(), default=1.0)
    candidates = [
        HybridCandidate(
            chunk=chunk,
            dense_score=dense_scores.get(chunk_id, 0.0),
            lexical_score=lexical_scores.get(chunk_id, 0.0),
            hybrid_score=(raw_hybrid_scores.get(chunk_id, 0.0) / max_hybrid_score)
            if max_hybrid_score
            else 0.0,
        )
        for chunk_id, chunk in chunks_by_id.items()
    ]
    candidates.sort(
        key=lambda item: (
            -item.hybrid_score,
            -item.lexical_score,
            -item.dense_score,
            item.chunk.page_start,
            item.chunk.chunk_index,
        )
    )
    return candidates[:candidate_limit]


def reciprocal_rank_score(rank: int | None, weight: float, rrf_k: int) -> float:
    if rank is None:
        return 0.0
    return weight / (rrf_k + rank)


def _ordered_anchor_pair_bonus(query: str, text: str) -> float:
    anchors = normalized_token_sequence(query, drop_stopwords=True)
    if len(anchors) < 2:
        return 0.0

    normalized_text = normalized_token_sequence(text, drop_stopwords=True)
    if len(normalized_text) < 2:
        return 0.0

    anchor_pairs = {
        (left, right)
        for left, right in zip(anchors, anchors[1:], strict=False)
        if left != right
    }
    if not anchor_pairs:
        return 0.0

    text_pairs = {
        (left, right)
        for left, right in zip(normalized_text, normalized_text[1:], strict=False)
    }
    pair_overlap = len(anchor_pairs & text_pairs)
    if pair_overlap == 0:
        return 0.0
    return min(0.2, pair_overlap / len(anchor_pairs) * 0.2)


def _query_term_rarity_bonus(
    *,
    query_terms: set[str],
    passage_terms: set[str],
    term_document_frequency: dict[str, int],
    candidate_count: int,
) -> float:
    matched_terms = query_terms & passage_terms
    if not matched_terms:
        return 0.0

    rarity_sum = 0.0
    for term in matched_terms:
        document_frequency = max(term_document_frequency.get(term, 0), 1)
        rarity_sum += log((candidate_count + 1) / document_frequency)

    normalization = max(len(query_terms), 1)
    return min(1.0, rarity_sum / normalization)


def rerank_matches(
    query: str,
    fused_candidates: list[HybridCandidate],
    top_k: int,
) -> list[ChunkSearchMatch]:
    if not fused_candidates:
        return []

    settings = get_settings()
    reranker = get_reranker()
    passages = [candidate.chunk.text for candidate in fused_candidates]
    try:
        rerank_scores = reranker.score(query, passages)
    except Exception:
        rerank_scores = TokenOverlapReranker().score(query, passages)
    matches = [
        ChunkSearchMatch(
            chunk=candidate.chunk,
            dense_score=candidate.dense_score,
            lexical_score=candidate.lexical_score,
            hybrid_score=candidate.hybrid_score,
            rerank_score=rerank_score,
            score=(candidate.hybrid_score * (1 - settings.retrieval_rerank_weight))
            + (rerank_score * settings.retrieval_rerank_weight),
        )
        for candidate, rerank_score in zip(fused_candidates, rerank_scores, strict=True)
    ]
    matches = [
        ChunkSearchMatch(
            chunk=match.chunk,
            dense_score=match.dense_score,
            lexical_score=match.lexical_score,
            hybrid_score=match.hybrid_score,
            rerank_score=match.rerank_score,
            score=max(
                0.0,
                match.score
                + min(
                    1.0,
                    lexical_overlap_score(
                        query=query,
                        text=match.chunk.text,
                        heading=match.chunk.heading,
                    ),
                )
                * 0.18
                + _focused_query_match_bonus(
                    query=query,
                    text=f"{match.chunk.heading or ''} {match.chunk.text}".strip(),
                )
                * 0.24
                + query_run_bonus(
                    query,
                    f"{match.chunk.heading or ''} {match.chunk.text}".strip(),
                    max_bonus=0.16,
                )
                - metadata_noise_score(match.chunk.text) * 0.25,
            ),
        )
        for match in matches
    ]
    matches.sort(
        key=lambda item: (
            -item.score,
            -item.rerank_score,
            -item.hybrid_score,
            -item.lexical_score,
            -item.dense_score,
            item.chunk.page_start,
            item.chunk.chunk_index,
        )
    )
    return matches[:top_k]


def _focused_query_match_bonus(*, query: str, text: str) -> float:
    query_terms = tokenize_for_search(query)
    if not query_terms:
        return 0.0

    passage_terms = tokenize_for_search(text)
    if not passage_terms:
        return 0.0

    anchor_terms = extract_anchor_terms(query)
    constraint_terms = extract_constraint_terms(query)
    matched_anchor_terms = anchor_terms & passage_terms
    matched_constraint_terms = constraint_terms & passage_terms

    bonus = 0.0
    if anchor_terms:
        bonus += len(matched_anchor_terms) / len(anchor_terms) * 0.45
    if constraint_terms:
        bonus += len(matched_constraint_terms) / len(constraint_terms) * 0.4
    if anchor_terms and matched_anchor_terms == anchor_terms:
        bonus += 0.08
    if constraint_terms and matched_constraint_terms == constraint_terms:
        bonus += 0.1

    if query.lower().startswith("when "):
        if YEAR_RE.search(text):
            bonus += 0.1
        if MONTH_RE.search(text):
            bonus += 0.1

    return min(1.0, bonus)
