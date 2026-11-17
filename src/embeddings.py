
import os
import time

from dotenv import load_dotenv
from google import genai
from google.genai import types
from google.genai import errors


load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    raise ValueError(
        "GEMINI_API_KEY not found in environment variables"
    )

client = genai.Client(api_key=api_key)

EMBEDDING_MODEL = "gemini-embedding-001"
BATCH_SIZE = 100
BATCH_DELAY = 65
MAX_RETRIES = 3


def call_embedding_api(contents, task_type):
    """Call Gemini embeddings with limited retries."""

    for attempt in range(MAX_RETRIES):
        try:
            return client.models.embed_content(
                model=EMBEDDING_MODEL,
                contents=contents,
                config=types.EmbedContentConfig(
                    task_type=task_type
                )
            )

        except errors.APIError:
            # Includes quota errors. Do not retry blindly.
            raise

        except Exception as error:
            if attempt == MAX_RETRIES - 1:
                raise

            delay = 2 ** attempt

            print(
                f"Embedding connection failed: {error}. "
                f"Retrying in {delay} seconds..."
            )

            time.sleep(delay)


def embed_documents(texts):
    """Generate document embeddings in batches."""

    if not texts:
        raise ValueError("No text provided for embedding")

    all_embeddings = []

    for start in range(0, len(texts), BATCH_SIZE):
        batch = texts[start:start + BATCH_SIZE]

        print(
            f"Processing batch: {start + 1} "
            f"to {start + len(batch)}"
        )

        response = call_embedding_api(
            contents=batch,
            task_type="RETRIEVAL_DOCUMENT"
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

        if start + BATCH_SIZE < len(texts):
            print(
                f"Waiting {BATCH_DELAY} seconds "
                "before the next batch..."
            )
            time.sleep(BATCH_DELAY)

    return all_embeddings


def embed_query(query):
    """Generate an embedding for a search query."""

    if not query or not query.strip():
        raise ValueError("Query cannot be empty")

    response = call_embedding_api(
        contents=query,
        task_type="RETRIEVAL_QUERY"
    )

    if not response.embeddings:
        raise ValueError("Gemini returned no query embedding")

    return response.embeddings[0].values
