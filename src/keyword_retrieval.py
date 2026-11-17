
import re

from rank_bm25 import BM25Okapi
from src.vector_store import get_collection


def tokenize(text):
    """
    Convert text into lowercase word tokens.
    """
    return re.findall(r"\b[a-zA-Z0-9]+\b", text.lower())


def search_keywords(
    query,
    n_results=5,
    filename=None
):
    """
    Retrieve document chunks using BM25 keyword ranking.

    Supports filtering by filename.
    """

    if not query or not query.strip():
        raise ValueError("Search query cannot be empty")

    collection = get_collection()

    # Retrieve stored documents and metadata
    stored_data = collection.get(
        where={"filename": filename} if filename else None,
        include=["documents", "metadatas"]
    )

    documents = stored_data["documents"]
    metadatas = stored_data["metadatas"]

    if not documents:
        return []

    # Tokenize all documents
    tokenized_documents = [
        tokenize(document)
        for document in documents
    ]

    # Build BM25 index
    bm25 = BM25Okapi(tokenized_documents)

    # Tokenize query
    tokenized_query = tokenize(query)

    if not tokenized_query:
        return []

    # Calculate BM25 scores
    scores = bm25.get_scores(tokenized_query)

    # Rank documents by score
    ranked_indices = sorted(
        range(len(scores)),
        key=lambda i: scores[i],
        reverse=True
    )

    results = []

    for index in ranked_indices:

        score = float(scores[index])

        # Skip documents with no keyword relevance
        if score <= 0:
            continue

        results.append({
            "text": documents[index],
            "metadata": metadatas[index],
            "keyword_score": score
        })

        if len(results) >= n_results:
            break

    return results

import re

from rank_bm25 import BM25Okapi
from src.vector_store import get_collection


def tokenize(text):
    """
    Convert text into lowercase word tokens.
    """
    return re.findall(r"\b[a-zA-Z0-9]+\b", text.lower())


def search_keywords(
    query,
    n_results=5,
    filename=None
):
    """
    Retrieve document chunks using BM25 keyword ranking.

    Supports filtering by filename.
    """

    if not query or not query.strip():
        raise ValueError("Search query cannot be empty")

    collection = get_collection()

    # Retrieve stored documents and metadata
    stored_data = collection.get(
        where={"filename": filename} if filename else None,
        include=["documents", "metadatas"]
    )

    documents = stored_data["documents"]
    metadatas = stored_data["metadatas"]

    if not documents:
        return []

    # Tokenize all documents
    tokenized_documents = [
        tokenize(document)
        for document in documents
    ]

    # Build BM25 index
    bm25 = BM25Okapi(tokenized_documents)

    # Tokenize query
    tokenized_query = tokenize(query)

    if not tokenized_query:
        return []

    # Calculate BM25 scores
    scores = bm25.get_scores(tokenized_query)

    # Rank documents by score
    ranked_indices = sorted(
        range(len(scores)),
        key=lambda i: scores[i],
        reverse=True
    )

    results = []

    for index in ranked_indices:

        score = float(scores[index])

        # Skip documents with no keyword relevance
        if score <= 0:
            continue

        results.append({
            "text": documents[index],
            "metadata": metadatas[index],
            "keyword_score": score
        })

        if len(results) >= n_results:
            break

    return results
