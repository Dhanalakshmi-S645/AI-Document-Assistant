
import hashlib
from pathlib import Path

from src.ingestion import process_pdf
from src.embeddings import embed_documents
from src.vector_store import get_collection


def get_file_hash(pdf_path):
    """Generate SHA-256 hash of PDF contents."""
    return hashlib.sha256(
        Path(pdf_path).read_bytes()
    ).hexdigest()


def store_pdf(pdf_path):
    """
    Store PDF chunks with content-based deduplication.

    Supports migration of older documents that do not
    have content_hash metadata.
    """

    pdf_path = Path(pdf_path)
    filename = pdf_path.name
    content_hash = get_file_hash(pdf_path)

    collection = get_collection()

    # -----------------------------------
    # 1. Check whether exact content exists
    # -----------------------------------

    existing_hash = collection.get(
        where={"content_hash": content_hash},
        include=["metadatas"],
        limit=1
    )

    if existing_hash["ids"]:
        print(
            f"Skipping {filename}: identical content already indexed."
        )
        return 0

    # -----------------------------------
    # 2. Extract PDF chunks
    # -----------------------------------

    chunks = process_pdf(pdf_path)

    if not chunks:
        raise ValueError("No readable text found in PDF")

    texts = [
        item["text"]
        for item in chunks
    ]

    # -----------------------------------
    # 3. Check legacy documents by filename
    # -----------------------------------

    legacy_data = collection.get(
        where={"filename": filename},
        include=["documents", "metadatas"]
    )

    if legacy_data["ids"]:

        old_documents = legacy_data["documents"]

        # Compare existing chunks with newly extracted chunks
        if old_documents == texts:

            # Same content: update metadata only.
            updated_metadatas = []

            for metadata in legacy_data["metadatas"]:
                updated_metadata = dict(metadata)
                updated_metadata["content_hash"] = content_hash

                updated_metadatas.append(updated_metadata)

            collection.update(
                ids=legacy_data["ids"],
                metadatas=updated_metadatas
            )

            print(
                f"Migrated legacy document: {filename}"
            )
            print(
                "Existing embeddings preserved. No Gemini request made."
            )

            return 0

    # -----------------------------------
    # 4. Generate embeddings for new content
    # -----------------------------------

    embeddings = embed_documents(texts)

    # -----------------------------------
    # 5. Generate chunk IDs
    # -----------------------------------

    ids = [
        f"{filename}-{content_hash[:12]}-page-{item['page']}-chunk-{item['chunk']}"
        for item in chunks
    ]

    # -----------------------------------
    # 6. Prepare metadata
    # -----------------------------------

    metadatas = [
        {
            "filename": filename,
            "page": item["page"],
            "chunk": item["chunk"],
            "section": item["section"],
            "content_hash": content_hash
        }
        for item in chunks
    ]

    # -----------------------------------
    # 7. Replace old document after embedding
    # -----------------------------------

    if legacy_data["ids"]:
        collection.delete(
            where={"filename": filename}
        )

    collection.upsert(
        ids=ids,
        documents=texts,
        embeddings=embeddings,
        metadatas=metadatas
    )

    print(f"Successfully indexed: {filename}")
    print(f"Chunks stored: {len(chunks)}")

    return len(chunks)


def store_all_pdfs(directory="documents"):
    """Process all PDFs in a directory."""

    pdf_files = list(
        Path(directory).glob("*.pdf")
    )

    if not pdf_files:
        print("No PDF files found.")
        return 0

    total_chunks = 0

    for pdf_path in pdf_files:

        print(f"\nProcessing: {pdf_path.name}")

        try:
            chunks_stored = store_pdf(pdf_path)
            total_chunks += chunks_stored

        except Exception as error:
            print(
                f"Failed to process {pdf_path.name}: {error}"
            )

    print(f"\nTotal new chunks stored: {total_chunks}")

    return total_chunks
