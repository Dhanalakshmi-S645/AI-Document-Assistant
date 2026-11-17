
import chromadb


# Initialize persistent ChromaDB client
client = chromadb.PersistentClient(
    path="./chroma_db"
)


# Create or retrieve our document collection
collection = client.get_or_create_collection(
    name="documents",
    metadata={"hnsw:space": "cosine"}
)


def get_collection():
    """
    Return the ChromaDB document collection.
    """
    return collection
