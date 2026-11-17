
import re

from rank_bm25 import BM25Okapi

from src.vector_store import get_collection


STOPWORDS = {
    "a", "an", "the", "is", "are", "was", "were",
    "what", "which", "who", "where", "when", "why",
    "how", "does", "do", "did", "and", "or", "of",
    "to", "in", "on", "for", "with", "by", "from",
    "it", "its", "this", "that", "these", "those",
    "my", "your", "their", "about", "discusses",
    "discuss", "mentioned", "mentions", "listed",
    "list", "explain", "describe", "purpose",
    "have", "has", "had", "i", "me", "we",
    "project", "professional"
}


PERSONAL_CONTEXT_TERMS = {
    "resume", "experience", "employment",
    "background", "career", "worked"
}



def tokenize(text):
    """
    Normalize text while preserving compound technical terms
    and making their individual components searchable.
    """

    text = text.lower()

    technical_terms = re.findall(
        r"\.net|c\+\+|c#",
        text
    )

    text = re.sub(
        r"\.net|c\+\+|c#",
        " ",
        text
    )

    raw_tokens = re.findall(
        r"[a-z0-9]+(?:[-_][a-z0-9]+)*",
        text
    )

    tokens = []

    for token in raw_tokens:

        # Preserve the complete compound term.
        tokens.append(token)

        # Also index its individual components.
        if "-" in token or "_" in token:

            components = re.split(r"[-_]", token)

            tokens.extend(
                component
                for component in components
                if component
            )

    tokens.extend(technical_terms)

    return tokens



def get_phrase_position(query_tokens, document_tokens):
    """
    Find the position of the meaningful query phrase
    inside a document.
    """

    if not query_tokens:
        return None

    phrase_length = len(query_tokens)

    for index in range(
        len(document_tokens) - phrase_length + 1
    ):

        if (
            document_tokens[index:index + phrase_length]
            == query_tokens
        ):
            return index

    return None


def get_document_boost(query_tokens, metadata, phrase_position):
    """
    Add contextual ranking signals without replacing BM25.
    """

    boost = 0.0

    filename = metadata.get("filename", "").lower()

    # Prefer resume documents for personal background questions.
    if PERSONAL_CONTEXT_TERMS.intersection(query_tokens):

        if "resume" in filename or "cv" in filename:
            boost += 8.0

    # Reward documents that introduce the queried phrase early.
    if phrase_position is not None:

        boost += 5.0 / (
            1.0 + phrase_position / 20.0
        )

    return boost


def search_keywords(
    query,
    n_results=5,
    filename=None
):
    """
    Retrieve document chunks using BM25.

    Combines BM25 relevance, query-term coverage,
    personal-document preference, and phrase-position
    signals.
    """

    if not query or not query.strip():
        raise ValueError("Query cannot be empty")

    if n_results < 1:
        raise ValueError("n_results must be positive")

    collection = get_collection()

    stored_data = collection.get(
        where={"filename": filename} if filename else None,
        include=["documents", "metadatas"]
    )

    documents = stored_data["documents"]
    metadatas = stored_data["metadatas"]

    if not documents:
        return []

    tokenized_documents = [
        tokenize(document)
        for document in documents
    ]

    tokenized_query = tokenize(query)

    if not tokenized_query:
        return []

    meaningful_query = [
        token
        for token in tokenized_query
        if token not in STOPWORDS
    ]

    if not meaningful_query:
        meaningful_query = tokenized_query

    bm25 = BM25Okapi(tokenized_documents)

    scores = bm25.get_scores(meaningful_query)

    query_terms = set(meaningful_query)

    ranked_results = []

    for index, document in enumerate(documents):

        document_tokens = tokenized_documents[index]

        document_token_set = set(document_tokens)

        matched_terms = query_terms.intersection(
            document_token_set
        )

        if not matched_terms:
            continue

        coverage = (
            len(matched_terms) / len(query_terms)
        )

        keyword_score = float(scores[index])

        phrase_position = get_phrase_position(
            meaningful_query,
            document_tokens
        )

        context_boost = get_document_boost(
            meaningful_query,
            metadatas[index],
            phrase_position
        )

        adjusted_score = (
            keyword_score
            + coverage
            + context_boost
        )

        ranked_results.append({
            "text": document,
            "metadata": metadatas[index],
            "keyword_score": keyword_score,
            "coverage": coverage,
            "context_boost": context_boost,
            "adjusted_score": adjusted_score
        })

    ranked_results.sort(
        key=lambda item: item["adjusted_score"],
        reverse=True
    )

    return ranked_results[:n_results]
