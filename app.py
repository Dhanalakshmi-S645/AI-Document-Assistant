
import streamlit as st
from pathlib import Path

from src.store import store_pdf
from src.retrieval import search_hybrid
from src.answerability import verify_evidence
from src.generation import generate_answer


st.set_page_config(
    page_title="AI Document Assistant",
    page_icon="📄",
    layout="wide"
)

st.title("📄 AI Document Assistant")

st.write(
    "Upload a PDF and ask questions about its contents. "
    "Answers are generated using retrieved document evidence."
)


# Initialize session state
if "uploaded_filename" not in st.session_state:
    st.session_state.uploaded_filename = None


# Sidebar: PDF upload
with st.sidebar:
    st.header("Document Upload")

    uploaded_file = st.file_uploader(
        "Upload your PDF",
        type=["pdf"]
    )

    if uploaded_file is not None:

        if uploaded_file.name != st.session_state.uploaded_filename:

            documents_dir = Path("documents")
            documents_dir.mkdir(exist_ok=True)

            pdf_path = documents_dir / uploaded_file.name

            try:
                with open(pdf_path, "wb") as file:
                    file.write(uploaded_file.getbuffer())

                with st.spinner("Processing PDF..."):
                    chunks_stored = store_pdf(str(pdf_path))

                st.session_state.uploaded_filename = uploaded_file.name

                st.success(
                    f"Document processed successfully. "
                    f"{chunks_stored} chunks stored."
                )

            except Exception as error:
                st.error(f"Document processing failed: {error}")

        else:
            st.success(f"Ready: {uploaded_file.name}")


# Main question interface
st.header("Ask your document")

question = st.text_input(
    "Enter your question",
    placeholder="Example: What is the educational qualification?"
)

ask_button = st.button(
    "Ask Question",
    type="primary"
)


if ask_button:

    if uploaded_file is None:
        st.warning("Please upload a PDF first.")

    elif st.session_state.uploaded_filename != uploaded_file.name:
        st.warning("Please wait until document processing completes.")

    elif not question.strip():
        st.warning("Please enter a question.")

    else:

        try:
            with st.spinner("Searching document evidence..."):

                # Step 1: Hybrid retrieval
                retrieved_chunks = search_hybrid(
                    query=question,
                    n_results=5,
                    vector_k=10,
                    keyword_k=10,
                    max_distance=0.40,
                    filename=uploaded_file.name
                )

                # Step 2: Verify retrieved evidence
                verification = verify_evidence(
                    question,
                    retrieved_chunks
                )

            st.subheader("Answer")

            decision = verification["decision"]

            if decision == "SUPPORTED":

                selected_evidence = [
                    retrieved_chunks[evidence_id - 1]
                    for evidence_id in verification["evidence_ids"]
                ]

                with st.spinner("Generating answer..."):
                    answer = generate_answer(
                        question,
                        selected_evidence,
                        verification
                    )

                st.write(answer)

            elif decision == "NOT_SUPPORTED":
                st.warning(
                    "I couldn't find sufficient supporting "
                    "information in the uploaded document."
                )

            elif decision == "CONTRADICTED":
                st.warning(
                    "The retrieved evidence contradicts "
                    "the claim in your question."
                )

            else:
                st.error(
                    "Evidence verification is temporarily unavailable. "
                    "Please try again."
                )

            # Evidence details
            with st.expander("View evidence verification details"):

                st.write("Decision:", decision)
                st.write("Reason:", verification["reason"])

                selected_ids = verification["evidence_ids"]

                if selected_ids:
                    st.subheader("Selected supporting evidence")

                    for evidence_id in selected_ids:
                        chunk = retrieved_chunks[evidence_id - 1]
                        metadata = chunk["metadata"]

                        st.success(
                            f"{metadata.get('filename')} "
                            f"— Page {metadata.get('page')}"
                        )

                        st.write(chunk["text"])

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

                        st.caption(
                            f"RRF score: {chunk['rrf_score']:.6f}"
                        )

                        st.write(chunk["text"])

        except Exception as error:
            st.error(
                f"An unexpected error occurred: {error}"
            )
