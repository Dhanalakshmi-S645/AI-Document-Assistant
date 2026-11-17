
import os
import hashlib
import streamlit as st
from pathlib import Path
from dotenv import load_dotenv


# -----------------------------------
# Environment configuration
# -----------------------------------

load_dotenv()

try:
    cloud_api_key = st.secrets["GEMINI_API_KEY"]
    os.environ["GEMINI_API_KEY"] = cloud_api_key
except (FileNotFoundError, KeyError):
    pass


# Import project modules after loading environment variables
from src.store import store_pdf
from src.retrieval import search_hybrid
from src.answerability import verify_evidence
from src.generation import generate_answer
from src.vector_store import get_collection


# -----------------------------------
# Streamlit configuration
# -----------------------------------

st.set_page_config(
    page_title="AI Document Assistant",
    page_icon="📄",
    layout="wide"
)

st.title("📄 AI Document Assistant")

st.write(
    "Upload multiple PDFs and ask questions across all your documents. "
    "Answers are generated using retrieved document evidence."
)

DOCUMENTS_DIR = Path("documents")
DOCUMENTS_DIR.mkdir(exist_ok=True)


# -----------------------------------
# Document upload and library
# -----------------------------------

with st.sidebar:

    st.header("Document Library")

    uploaded_files = st.file_uploader(
        "Upload PDF documents",
        type=["pdf"],
        accept_multiple_files=True
    )

    if uploaded_files:

        for uploaded_file in uploaded_files:

            filename = Path(uploaded_file.name).name
            pdf_path = DOCUMENTS_DIR / filename

            try:

                file_bytes = uploaded_file.getvalue()

                uploaded_hash = hashlib.sha256(
                    file_bytes
                ).hexdigest()

                existing_hash = None

                if pdf_path.exists():
                    existing_hash = hashlib.sha256(
                        pdf_path.read_bytes()
                    ).hexdigest()

                if existing_hash != uploaded_hash:
                    pdf_path.write_bytes(file_bytes)

                with st.spinner(f"Checking {filename}..."):

                    chunks_stored = store_pdf(
                        str(pdf_path)
                    )

                if chunks_stored > 0:

                    st.success(
                        f"{filename}: {chunks_stored} chunks indexed."
                    )

                else:

                    st.info(
                        f"{filename}: identical content already indexed."
                    )

            except Exception as error:

                st.error(
                    f"Failed to process {filename}: {error}"
                )


    # -----------------------------------
    # Display indexed documents
    # -----------------------------------

    collection = get_collection()

    stored_data = collection.get(
        include=["metadatas"]
    )

    stored_filenames = sorted({
        metadata["filename"]
        for metadata in stored_data["metadatas"]
        if metadata and metadata.get("filename")
    })

    st.divider()
    st.subheader("Processed Documents")

    if stored_filenames:

        for filename in stored_filenames:
            st.write(f"📄 {filename}")

        st.caption(
            f"{len(stored_filenames)} document(s) indexed"
        )

    else:

        st.info("No documents uploaded yet.")


# -----------------------------------
# Question answering
# -----------------------------------

st.header("Ask Across All Documents")

question = st.text_input(
    "Enter your question",
    placeholder=(
        "Example: What programming languages "
        "are listed in my resume?"
    )
)

ask_button = st.button(
    "Ask Question",
    type="primary"
)


if ask_button:

    if not question.strip():

        st.warning("Please enter a question.")

    elif get_collection().count() == 0:

        st.warning("Please upload at least one PDF.")

    else:

        try:

            # -----------------------------------
            # Hybrid retrieval
            # -----------------------------------

            with st.spinner("Searching across all documents..."):

                retrieved_chunks = search_hybrid(
                    query=question,
                    n_results=5,
                    vector_k=10,
                    keyword_k=10,
                    max_distance=0.40,
                    filename=None,
                    max_per_document=2
                )

            if not retrieved_chunks:

                st.warning(
                    "No relevant evidence was found "
                    "in the indexed documents."
                )

                st.stop()


            # -----------------------------------
            # Evidence verification
            # -----------------------------------

            with st.spinner("Verifying retrieved evidence..."):

                verification = verify_evidence(
                    question,
                    retrieved_chunks
                )

            st.subheader("Answer")

            decision = verification["decision"]


            # -----------------------------------
            # Generate answer for supported questions
            # -----------------------------------

            if decision == "SUPPORTED":

                selected_evidence = [
                    retrieved_chunks[evidence_id - 1]
                    for evidence_id in verification["evidence_ids"]
                    if (
                        isinstance(evidence_id, int)
                        and 1 <= evidence_id <= len(retrieved_chunks)
                    )
                ]

                if not selected_evidence:

                    st.warning(
                        "No valid supporting evidence was selected."
                    )

                else:

                    with st.spinner("Generating grounded answer..."):

                        answer = generate_answer(
                            question,
                            selected_evidence,
                            verification
                        )

                    st.write(answer)


            # -----------------------------------
            # Not supported
            # -----------------------------------

            elif decision == "NOT_SUPPORTED":

                st.warning(
                    "I couldn't find sufficient supporting "
                    "information in the uploaded documents."
                )


            # -----------------------------------
            # Contradicted
            # -----------------------------------

            elif decision == "CONTRADICTED":

                st.warning(
                    "The retrieved evidence contradicts "
                    "the claim in your question."
                )


            # -----------------------------------
            # Verification error
            # -----------------------------------

            else:

                st.error(
                    "Evidence verification is temporarily unavailable. "
                    "Please try again."
                )


            # -----------------------------------
            # Evidence verification details
            # -----------------------------------

            with st.expander("View evidence verification details"):

                st.write("Decision:", decision)

                st.write(
                    "Reason:",
                    verification.get("reason", "No reason provided.")
                )

                selected_ids = verification.get(
                    "evidence_ids", []
                )


                # -----------------------------------
                # Selected supporting evidence
                # -----------------------------------

                if selected_ids:

                    st.subheader("Selected Supporting Evidence")

                    for evidence_id in selected_ids:

                        if not (
                            isinstance(evidence_id, int)
                            and 1 <= evidence_id <= len(retrieved_chunks)
                        ):
                            continue

                        chunk = retrieved_chunks[evidence_id - 1]
                        metadata = chunk["metadata"]

                        st.success(
                            f"{metadata.get('filename')} "
                            f"— Page {metadata.get('page')}"
                        )

                        st.write(chunk["text"])


                # -----------------------------------
                # All retrieved candidates
                # -----------------------------------

                with st.expander("View all retrieved candidates"):

                    for index, chunk in enumerate(
                        retrieved_chunks,
                        start=1
                    ):

                        metadata = chunk["metadata"]

                        st.markdown(
                            f"**Evidence {index}** — "
                            f"{metadata.get('filename')} "
                            f"(Page {metadata.get('page')})"
                        )

                        if "rrf_score" in chunk:

                            st.caption(
                                f"RRF score: {chunk['rrf_score']:.6f}"
                            )

                        elif "adjusted_score" in chunk:

                            st.caption(
                                "BM25 adjusted score: "
                                f"{chunk['adjusted_score']:.6f}"
                            )

                        else:

                            st.caption("Retrieval score unavailable.")

                        st.write(chunk["text"])

                        st.divider()


        except Exception as error:

            st.error(
                f"An unexpected error occurred: {error}"
            )
