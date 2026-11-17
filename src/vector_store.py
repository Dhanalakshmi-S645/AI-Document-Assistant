
import chromadb


client = chromadb.PersistentClient(
    path="./chroma_db"
)

collection = client.get_or_create_collection(
    name="documents_minilm",
    metadata={
        "hnsw:space": "cosine",
        "embedding_model": "all-MiniLM-L6-v2"
    }
)


def get_collection():
    """Return the document collection."""
    return collection
