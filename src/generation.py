
from google.genai import errors
from src.embeddings import client


GENERATION_MODEL = "gemini-3.5-flash-lite"


def generate_answer(query, retrieved_chunks, verification=None):
    """
    Generate an evidence-grounded answer.

    The verifier's decision is used as a gate, but the
    generator must still independently follow the evidence.
    """

    if not retrieved_chunks:
        return (
            "I couldn't find sufficient supporting information "
            "in the uploaded documents."
        )

    if verification is not None:
        if verification.get("decision") != "SUPPORTED":
            return (
                "I couldn't find sufficient supporting information "
                "in the uploaded documents."
            )

    context_parts = []

    for index, chunk in enumerate(retrieved_chunks, start=1):
        metadata = chunk["metadata"]

        source = (
            f"[Evidence ID: {index}]\n"
            f"Filename: {metadata['filename']}\n"
            f"Page: {metadata['page']}\n"
            f"Section: {metadata.get('section', 'DOCUMENT')}"
        )

        context_parts.append(
            f"{source}\nContent:\n{chunk['text']}"
        )

    context = "\n\n".join(context_parts)

    verification_reason = (
        verification.get("reason", "")
        if verification else ""
    )

    prompt = f"""
You are a document question-answering assistant.

The evidence verification stage has classified the question
as SUPPORTED.

Verification reason:
{verification_reason}

Answer the user's question using only the document evidence below.

RULES:
1. Identify the evidence that directly answers the question.
2. State the answer explicitly and concisely.
3. Do not refuse when the evidence clearly contains the answer.
4. Do not add facts that are absent from the evidence.
5. Do not confuse information from different documents.
6. Cite the exact filename and page of the supporting evidence.
7. If the evidence genuinely does not contain the answer,
   state that sufficient information was not found.

CITATION FORMAT:
[Source: filename, Page: page_number]

DOCUMENT EVIDENCE:
{context}

USER QUESTION:
{query}

ANSWER:
"""

    try:
        response = client.models.generate_content(
            model=GENERATION_MODEL,
            contents=prompt
        )

        if not response.text:
            return "The AI model returned an empty response. Please try again."

        return response.text

    except errors.APIError as error:
        print(f"Gemini API error: {error}")
        return "The AI service is temporarily unavailable. Please try again."

    except Exception as error:
        print(f"Unexpected generation error: {error}")
        return "An unexpected error occurred while generating the answer."
