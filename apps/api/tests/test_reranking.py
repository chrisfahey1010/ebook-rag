from ebook_rag_api.services.reranking import TokenOverlapReranker


def test_token_overlap_reranker_prefers_passages_with_stronger_query_coverage() -> None:
    reranker = TokenOverlapReranker()

    scores = reranker.score(
        "battery maintenance schedule",
        [
            "Maintenance notes for battery systems and charging schedule.",
            "General overview of the spacecraft cabin layout.",
        ],
    )

    assert scores[0] > scores[1]
    assert scores[0] > 0
    assert scores[1] >= 0
