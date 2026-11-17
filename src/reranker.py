
from sentence_transformers import CrossEncoder


RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

reranker = CrossEncoder(
    RERANKER_MODEL,
    device="cpu"
)


# Maps query concepts to actual document sections.
SECTION_RULES = {
    "EDUCATION": [
        "degree",
        "college",
        "university",
        "education",
        "graduated",
        "engineering studies"
    ],

    "MIRA -Medical Intelligence Robotic Automation": [
        "mira",
        "diabetes application",
        "health application",
        "patient data"
    ],

    "Food Demand Forecasting-ML": [
        "food demand",
        "food forecasting",
        "meal orders",
        "forecasting model",
        "optuna",
        "rmsle",
        "feature engineering"
    ],

    "SKILLS": [
        "programming languages",
        "technical skills",
        "technologies known",
        "what skills",
        "which skills"
    ],

    "predictwise-ai": [
        "predictive maintenance",
        "machine failure",
        "equipment failure"
    ]
}


SECTION_BOOST = 0.0


def get_section_boost(query, metadata):
    """
    Apply a small relevance boost when the query
    matches the chunk's actual section metadata.
    """

    query_lower = query.lower()

    section = metadata.get("section", "")

    for target_section, keywords in SECTION_RULES.items():

        query_matches_section = any(
            keyword in query_lower
            for keyword in keywords
        )

        if (
            query_matches_section
            and section == target_section
        ):
            return SECTION_BOOST

    return 0.0


def rerank_documents(query, retrieved_chunks, top_k=3):

    if not query or not query.strip():
        raise ValueError("Query cannot be empty")

    if top_k < 1:
        raise ValueError("top_k must be at least 1")

    if not retrieved_chunks:
        return []

    # Prepare query-document pairs
    pairs = [
        [query, chunk["text"]]
        for chunk in retrieved_chunks
    ]

    # Cross-encoder relevance scores
    scores = reranker.predict(pairs)

    reranked_chunks = []

    for chunk, score in zip(retrieved_chunks, scores):

        chunk_copy = chunk.copy()

        cross_encoder_score = float(score)

        section_boost = get_section_boost(
            query,
            chunk["metadata"]
        )

        final_score = (
            cross_encoder_score + section_boost
        )

        chunk_copy["rerank_score"] = cross_encoder_score
        chunk_copy["section_boost"] = section_boost
        chunk_copy["final_score"] = final_score

        reranked_chunks.append(chunk_copy)

    # Higher scores indicate better relevance
    reranked_chunks.sort(
        key=lambda item: item["final_score"],
        reverse=True
    )

    return reranked_chunks[:top_k]
