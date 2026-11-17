import hashlib
import os
import time

from dotenv import load_dotenv
from google import genai
from google.genai import types


load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")
print(
    "GEMINI_KEY_FINGERPRINT:",
    hashlib.sha256(api_key.encode()).hexdigest()[:12]
    if api_key else "MISSING"
)
if not api_key:
    raise ValueError(
        "GEMINI_API_KEY not found in environment variables"
    )

client = genai.Client(api_key=api_key)

EMBEDDING_MODEL = "gemini-embedding-001"
BATCH_SIZE = 100
BATCH_DELAY = 65


def embed_documents(texts):
    """
    Generate document embeddings in batches while
    respecting the Gemini free-tier rate limit.
    """

    if not texts:
        raise ValueError("No text provided for embedding")

    all_embeddings = []

    for start in range(0, len(texts), BATCH_SIZE):

        batch = texts[start:start + BATCH_SIZE]

        print(
            f"Processing batch: {start + 1} "
            f"to {start + len(batch)}"
        )

        response = client.models.embed_content(
            model=EMBEDDING_MODEL,
            contents=batch,
            config=types.EmbedContentConfig(
                task_type="RETRIEVAL_DOCUMENT"
            )
        )

        batch_embeddings = [
            item.values
            for item in response.embeddings
        ]

        if len(batch_embeddings) != len(batch):
            raise ValueError(
                "Embedding count does not match input batch size"
            )

        all_embeddings.extend(batch_embeddings)

        print(
            f"Embedded {len(all_embeddings)}/{len(texts)} chunks"
        )

        # Wait before sending the next batch
        if start + BATCH_SIZE < len(texts):
            print(
                f"Waiting {BATCH_DELAY} seconds "
                "for the API rate limit to reset..."
            )
            time.sleep(BATCH_DELAY)

    return all_embeddings


def embed_query(query):
    """
    Generate an embedding for a search query.
    """

    if not query or not query.strip():
        raise ValueError("Query cannot be empty")

    response = client.models.embed_content(
        model=EMBEDDING_MODEL,
        contents=query,
        config=types.EmbedContentConfig(
            task_type="RETRIEVAL_QUERY"
        )
    )

    return response.embeddings[0].values
