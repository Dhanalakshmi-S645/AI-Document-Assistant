
from src.embeddings import embed_query
from src.vector_store import get_collection
from src.keyword_retrieval import search_keywords


def search_documents(
    query,
    n_results=3,
    max_distance=0.40,
    filename=None
):
    """
    Retrieve document chunks using semantic vector search.
    """

    if not query or not query.strip():
        raise ValueError("Search query cannot be empty")

    collection = get_collection()
    total_documents = collection.count()

    if total_documents == 0:
        return []

    query_embedding = embed_query(query)

    query_filter = (
        {"filename": filename}
        if filename
        else None
    )

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=min(n_results, total_documents),
        where=query_filter,
        include=["documents", "metadatas", "distances"]
    )

    retrieved_chunks = []

    for i in range(len(results["documents"][0])):

        distance = results["distances"][0][i]

        if distance > max_distance:
            continue

        retrieved_chunks.append({
            "text": results["documents"][0][i],
            "metadata": results["metadatas"][0][i],
            "distance": distance
        })

    return retrieved_chunks


def search_hybrid(
    query,
    n_results=5,
    vector_k=10,
    keyword_k=10,
    max_distance=0.40,
    filename=None,
    rrf_k=60
):
    """
    Hybrid retrieval using vector search, BM25
    and Reciprocal Rank Fusion.
    """

    if not query or not query.strip():
        raise ValueError("Search query cannot be empty")

    if n_results < 1 or vector_k < 1 or keyword_k < 1:
        raise ValueError("Retrieval limits must be positive")

    if rrf_k < 1:
        raise ValueError("rrf_k must be positive")

    # Retrieve independently
    vector_results = search_documents(
        query=query,
        n_results=vector_k,
        max_distance=max_distance,
        filename=filename
    )

    keyword_results = search_keywords(
        query=query,
        n_results=keyword_k,
        filename=filename
    )

    rrf_scores = {}
    documents = {}

    # Combine rankings using Reciprocal Rank Fusion
    for ranking in (vector_results, keyword_results):

        for rank, item in enumerate(ranking, start=1):

            metadata = item["metadata"]

            document_id = (
                metadata["filename"],
                metadata["page"],
                metadata["chunk"]
            )

            rrf_scores[document_id] = (
                rrf_scores.get(document_id, 0.0)
                + 1.0 / (rrf_k + rank)
            )

            # Preserve the original chunk and its metadata
            if document_id not in documents:
                documents[document_id] = item.copy()

    # Sort by combined RRF score
    ranked_results = sorted(
        rrf_scores.items(),
        key=lambda item: item[1],
        reverse=True
    )

    hybrid_results = []

    for document_id, score in ranked_results[:n_results]:

        chunk = documents[document_id].copy()

        chunk["rrf_score"] = score

        hybrid_results.append(chunk)

    return hybrid_results
