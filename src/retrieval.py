
from src.embeddings import embed_query
from src.vector_store import get_collection
from src.keyword_retrieval import search_keywords


def get_document_id(item):
    """Create a stable identifier for a retrieved chunk."""

    metadata = item["metadata"]

    return (
        metadata["filename"],
        metadata["page"],
        metadata["chunk"]
    )


def diversify_results(
    results,
    n_results=5,
    max_per_document=3
):
    """
    Reduce duplicate chunks and prevent one document
    from dominating the final evidence set.
    """

    diversified = []
    seen_chunks = set()
    document_counts = {}

    for item in results:

        document_id = get_document_id(item)

        filename = document_id[0]

        # Avoid duplicate chunks
        if document_id in seen_chunks:
            continue

        # Limit chunks from a single document
        if document_counts.get(filename, 0) >= max_per_document:
            continue

        seen_chunks.add(document_id)

        document_counts[filename] = (
            document_counts.get(filename, 0) + 1
        )

        diversified.append(item)

        if len(diversified) >= n_results:
            break

    return diversified


def search_documents(
    query,
    n_results=3,
    max_distance=0.40,
    filename=None
):
    """Retrieve document chunks using semantic vector search."""

    if not query or not query.strip():
        raise ValueError("Search query cannot be empty")

    if n_results < 1:
        raise ValueError("n_results must be positive")

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
        include=[
            "documents",
            "metadatas",
            "distances"
        ]
    )

    retrieved_chunks = []

    if not results["documents"] or not results["documents"][0]:
        return []

    for i in range(len(results["documents"][0])):

        distance = results["distances"][0][i]

        if distance > max_distance:
            continue

        retrieved_chunks.append({
            "text": results["documents"][0][i],
            "metadata": results["metadatas"][0][i],
            "distance": distance,
            "retrieval_method": ["Vector"]
        })

    return retrieved_chunks


def search_hybrid(
    query,
    n_results=5,
    vector_k=10,
    keyword_k=10,
    max_distance=0.40,
    filename=None,
    rrf_k=60,
    max_per_document=3
):
    """
    Hybrid retrieval using Reciprocal Rank Fusion.

    Falls back to BM25 when vector retrieval is unavailable.
    Applies document diversity to the final results.
    """

    if not query or not query.strip():
        raise ValueError("Search query cannot be empty")

    if n_results < 1 or vector_k < 1 or keyword_k < 1:
        raise ValueError("Retrieval limits must be positive")

    if rrf_k < 1:
        raise ValueError("rrf_k must be positive")

    if max_per_document < 1:
        raise ValueError("max_per_document must be positive")

    # -----------------------------------
    # 1. BM25 retrieval
    # -----------------------------------

    keyword_results = search_keywords(
        query=query,
        n_results=keyword_k,
        filename=filename
    )

    # -----------------------------------
    # 2. Semantic retrieval
    # -----------------------------------

    try:

        vector_results = search_documents(
            query=query,
            n_results=vector_k,
            max_distance=max_distance,
            filename=filename
        )

    except Exception as error:

        print(
            "Vector retrieval unavailable. "
            f"Using BM25 results only. Error: {error}"
        )

        fallback_results = []

        for item in keyword_results:

            result = item.copy()

            result["retrieval_method"] = ["BM25"]
            result["rrf_score"] = None

            fallback_results.append(result)

        return diversify_results(
            fallback_results,
            n_results=n_results,
            max_per_document=max_per_document
        )

    # -----------------------------------
    # 3. Handle empty results
    # -----------------------------------

    if not vector_results and not keyword_results:
        return []

    if not vector_results:

        fallback_results = []

        for item in keyword_results:

            result = item.copy()

            result["retrieval_method"] = ["BM25"]
            result["rrf_score"] = None

            fallback_results.append(result)

        return diversify_results(
            fallback_results,
            n_results=n_results,
            max_per_document=max_per_document
        )

    # -----------------------------------
    # 4. Reciprocal Rank Fusion
    # -----------------------------------

    rrf_scores = {}
    documents = {}
    retrieval_methods = {}

    for method, ranking in (
        ("Vector", vector_results),
        ("BM25", keyword_results)
    ):

        for rank, item in enumerate(ranking, start=1):

            document_id = get_document_id(item)

            rrf_scores[document_id] = (
                rrf_scores.get(document_id, 0.0)
                + 1.0 / (rrf_k + rank)
            )

            retrieval_methods.setdefault(
                document_id, []
            ).append(method)

            if document_id not in documents:
                documents[document_id] = item.copy()

    # -----------------------------------
    # 5. Rank combined results
    # -----------------------------------

    ranked_results = sorted(
        rrf_scores.items(),
        key=lambda item: item[1],
        reverse=True
    )

    hybrid_results = []

    for document_id, score in ranked_results:

        chunk = documents[document_id].copy()

        chunk["rrf_score"] = score

        chunk["retrieval_method"] = list(
            dict.fromkeys(retrieval_methods[document_id])
        )

        hybrid_results.append(chunk)

    # -----------------------------------
    # 6. Apply document diversity
    # -----------------------------------

    return diversify_results(
        hybrid_results,
        n_results=n_results,
        max_per_document=max_per_document
    )
