
from sentence_transformers import SentenceTransformer


# Local embedding model
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

# Load model once and reuse it
embedding_model = SentenceTransformer(EMBEDDING_MODEL)


def embed_documents(texts):
    """Generate local embeddings for document chunks."""

    if not texts:
        raise ValueError("No text provided for embedding")

    embeddings = embedding_model.encode(
        texts,
        batch_size=32,
        show_progress_bar=False,
        normalize_embeddings=True
    )

    return embeddings.tolist()


def embed_query(query):
    """Generate a local embedding for a search query."""

    if not query or not query.strip():
        raise ValueError("Query cannot be empty")

    embedding = embedding_model.encode(
        query,
        normalize_embeddings=True
    )

    return embedding.tolist()
