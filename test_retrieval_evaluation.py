
from src.retrieval import search_documents
from src.reranker import rerank_documents


test_cases = [
    {
        "category": "Direct factual",
        "question": "Which degree did the candidate complete?",
        "relevant_ids": ["resume.pdf-page-1-chunk-5"]
    },
    {
        "category": "Paraphrased",
        "question": "Where did the candidate pursue engineering studies?",
        "relevant_ids": ["resume.pdf-page-1-chunk-5"]
    },
    {
        "category": "Direct factual",
        "question": "Which database powers the MIRA application?",
        "relevant_ids": [
            "resume.pdf-page-1-chunk-3",
            "sample.pdf-page-1-chunk-1"
        ]
    },
    {
        "category": "Multi-detail",
        "question": "How was the Food Demand Forecasting model optimized?",
        "relevant_ids": [
            "resume.pdf-page-1-chunk-2",
            "sample.pdf-page-1-chunk-1"
        ]
    },
    {
        "category": "Multi-detail",
        "question": "Which feature engineering techniques were used for food demand prediction?",
        "relevant_ids": ["resume.pdf-page-1-chunk-2"]
    },
    {
        "category": "Unanswerable",
        "question": "What is the candidate's date of birth?",
        "relevant_ids": []
    },
    {
        "category": "Unanswerable",
        "question": "What was the candidate's previous monthly salary?",
        "relevant_ids": []
    }
]


def get_chunk_id(chunk):
    metadata = chunk["metadata"]

    return (
        f"{metadata['filename']}-"
        f"page-{metadata['page']}-"
        f"chunk-{metadata['chunk']}"
    )


def precision_at_k(results, relevant_ids, k=3):
    relevant_count = sum(
        get_chunk_id(chunk) in relevant_ids
        for chunk in results[:k]
    )

    # Fixed-k precision: missing results count as non-relevant.
    return relevant_count / k


def recall_at_k(results, relevant_ids, k=3):
    if not relevant_ids:
        return None

    relevant_count = sum(
        get_chunk_id(chunk) in relevant_ids
        for chunk in results[:k]
    )

    return relevant_count / len(relevant_ids)


def reciprocal_rank_at_k(results, relevant_ids, k=3):
    for rank, chunk in enumerate(results[:k], start=1):
        if get_chunk_id(chunk) in relevant_ids:
            return 1 / rank

    return 0.0


def evaluate_ranking(results, relevant_ids):
    return {
        "precision": precision_at_k(results, relevant_ids, k=3),
        "recall": recall_at_k(results, relevant_ids, k=3),
        "mrr": reciprocal_rank_at_k(results, relevant_ids, k=3)
    }


def calculate_average(metrics, key):
    if not metrics:
        return 0.0

    return sum(item[key] for item in metrics) / len(metrics)


def print_metrics(title, metrics):
    print(f"\n{title}")
    print(f"Precision@3: {metrics['precision']:.3f}")
    print(f"Recall@3:    {metrics['recall']:.3f}")
    print(f"MRR@3:       {metrics['mrr']:.3f}")


initial_metrics = []
reranked_metrics = []


for index, test in enumerate(test_cases, start=1):

    query = test["question"]
    relevant_ids = set(test["relevant_ids"])

    print("\n" + "=" * 80)
    print(f"TEST {index}: {test['category']}")
    print("QUESTION:", query)

    # Initial vector retrieval
    retrieved = search_documents(
        query,
        n_results=5,
        max_distance=0.40
    )

    # CrossEncoder reranking
    reranked = rerank_documents(
        query,
        retrieved,
        top_k=3
    )

    print("\nInitial retrieval:", len(retrieved))
    print("After reranking:", len(reranked))

    if relevant_ids:

        initial = evaluate_ranking(retrieved, relevant_ids)
        reranked_result = evaluate_ranking(reranked, relevant_ids)

        initial_metrics.append(initial)
        reranked_metrics.append(reranked_result)

        print_metrics("Initial retrieval:", initial)
        print_metrics("CrossEncoder reranking:", reranked_result)

    else:
        print("\nUnanswerable question.")
        print("Relevant chunks expected: 0")
        print("Retrieved chunks:", len(retrieved))

    print("\nInitial ranking:")

    for rank, chunk in enumerate(retrieved, start=1):
        print(
            f"{rank}. {get_chunk_id(chunk)} "
            f"| Distance: {chunk['distance']:.4f}"
        )

    print("\nCrossEncoder ranking:")

    for rank, chunk in enumerate(reranked, start=1):
        print(
            f"{rank}. {get_chunk_id(chunk)} "
            f"| CrossEncoder: {chunk['rerank_score']:.4f} "
            f"| Section boost: {chunk['section_boost']:.1f} "
            f"| Final: {chunk['final_score']:.4f}"
        )


# Overall evaluation
print("\n" + "=" * 80)
print("OVERALL RETRIEVAL EVALUATION")

if initial_metrics and reranked_metrics:

    print("\nInitial retrieval averages:")

    for metric in ["precision", "recall", "mrr"]:
        print(
            f"{metric.upper()}: "
            f"{calculate_average(initial_metrics, metric):.3f}"
        )

    print("\nCrossEncoder reranking averages:")

    for metric in ["precision", "recall", "mrr"]:
        print(
            f"{metric.upper()}: "
            f"{calculate_average(reranked_metrics, metric):.3f}"
        )

print("\nAnswerable test cases:", len(initial_metrics))
print("Unanswerable test cases:", len(test_cases) - len(initial_metrics))
print("Total test cases:", len(test_cases))

print("\nEvaluation completed.")
