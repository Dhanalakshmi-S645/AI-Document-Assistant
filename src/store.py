
from pathlib import Path

from src.ingestion import process_pdf
from src.embeddings import embed_documents
from src.vector_store import get_collection


def store_pdf(pdf_path):
    """
    Extract, embed and store PDF chunks with metadata.
    """

    chunks = process_pdf(pdf_path)

    if not chunks:
        raise ValueError("No readable text found in PDF")

    # Prepare chunk text
    texts = [
        item["text"]
        for item in chunks
    ]

    # Generate embeddings
    embeddings = embed_documents(texts)

    filename = Path(pdf_path).name

    # Generate unique IDs
    ids = [
        f"{filename}-page-{item['page']}-chunk-{item['chunk']}"
        for item in chunks
    ]

    # Store document metadata, including section
    metadatas = [
        {
            "filename": filename,
            "page": item["page"],
            "chunk": item["chunk"],
            "section": item["section"]
        }
        for item in chunks
    ]

    collection = get_collection()

    # Remove previous chunks for this document
    collection.delete(
        where={"filename": filename}
    )

    # Insert updated chunks
    collection.upsert(
        ids=ids,
        documents=texts,
        embeddings=embeddings,
        metadatas=metadatas
    )

    return len(chunks)


def store_all_pdfs(directory="documents"):
    """
    Process and store every PDF in the given directory.
    """

    pdf_files = list(
        Path(directory).glob("*.pdf")
    )

    if not pdf_files:
        print("No PDF files found.")
        return 0

    total_chunks = 0

    for pdf_path in pdf_files:

        print(f"\nProcessing: {pdf_path.name}")

        chunks_stored = store_pdf(pdf_path)

        total_chunks += chunks_stored

        print(f"Stored chunks: {chunks_stored}")

    return total_chunks
