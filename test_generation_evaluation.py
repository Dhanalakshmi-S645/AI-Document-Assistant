
from src.retrieval import search_documents
from src.reranker import rerank_documents
from src.generation import generate_answer


test_questions = [
    "Which degree did the candidate complete?",
    "Where did the candidate pursue engineering studies?",
    "Which database powers the MIRA application?",
    "How was the Food Demand Forecasting model optimized?",
    "Which feature engineering techniques were used for food demand prediction?",
    "What is the candidate's date of birth?",
    "What was the candidate's previous monthly salary?"
]


for index, query in enumerate(test_questions, start=1):

    print("\n" + "=" * 75)
    print(f"TEST {index}")
    print("QUESTION:", query)

    # Retrieve candidate chunks
    retrieved_chunks = search_documents(
        query,
        n_results=5,
        max_distance=0.40
    )

    # Rerank the retrieved chunks
    reranked_chunks = rerank_documents(
        query,
        retrieved_chunks,
        top_k=3
    )

    # Generate an answer from the reranked evidence
    answer = generate_answer(
        query,
        reranked_chunks
    )

    print("\nRetrieved:", len(retrieved_chunks))
    print("Reranked:", len(reranked_chunks))

    print("\nGENERATED ANSWER:")
    print(answer)

print("\n" + "=" * 75)
print("GENERATION EVALUATION COMPLETED")
