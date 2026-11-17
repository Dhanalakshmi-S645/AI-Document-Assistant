
from src.retrieval import search_documents
from src.answerability import verify_evidence


# Ground-truth evaluation dataset
TEST_CASES = [
    {
        "query": "Which degree did the candidate complete?",
        "expected": True
    },
    {
        "query": "Where did the candidate pursue engineering studies?",
        "expected": True
    },
    {
        "query": "Which database powers MIRA?",
        "expected": True
    },
    {
        "query": "How was the Food Demand Forecasting model optimized?",
        "expected": True
    },
    {
        "query": "Which feature engineering techniques were used for food demand prediction?",
        "expected": True
    },
    {
        "query": "Which ML library was used for food demand forecasting?",
        "expected": True
    },
    {
        "query": "Which algorithms were used in the predictive maintenance project?",
        "expected": True
    },
    {
        "query": "What is the candidate's date of birth?",
        "expected": False
    },
    {
        "query": "What was the candidate's previous monthly salary?",
        "expected": False
    },
    {
        "query": "Which cloud provider hosts the candidate's projects?",
        "expected": False
    },
    {
        "query": "How many years of full-time ML experience does the candidate have?",
        "expected": False
    },
    {
        "query": "Which database was used in Food Demand Forecasting?",
        "expected": False
    }
]


def calculate_metrics(results):

    # API errors are excluded from classification metrics.
    valid_results = [
        item for item in results
        if item["decision"] != "ERROR"
    ]

    tp = tn = fp = fn = 0

    for item in valid_results:

        expected = item["expected"]
        predicted = item["predicted"]

        if expected and predicted:
            tp += 1
        elif not expected and not predicted:
            tn += 1
        elif not expected and predicted:
            fp += 1
        elif expected and not predicted:
            fn += 1

    total = len(valid_results)

    accuracy = (
        (tp + tn) / total
        if total else 0
    )

    precision = (
        tp / (tp + fp)
        if (tp + fp) else 0
    )

    recall = (
        tp / (tp + fn)
        if (tp + fn) else 0
    )

    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) else 0
    )

    return {
        "TP": tp,
        "TN": tn,
        "FP": fp,
        "FN": fn,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "valid": total,
        "errors": len(results) - total
    }


def run_evaluation():

    results = []

    print("\n" + "=" * 65)
    print("ANSWERABILITY EVALUATION")
    print("=" * 65)

    for index, test in enumerate(TEST_CASES, start=1):

        query = test["query"]
        expected = test["expected"]

        print("\n" + "-" * 65)
        print(f"TEST {index}/{len(TEST_CASES)}")
        print("QUESTION:", query)
        print(
            "EXPECTED:",
            "SUPPORTED" if expected else "NOT_SUPPORTED"
        )

        # Step 1: Initial retrieval
        retrieved = search_documents(
            query,
            n_results=5,
            max_distance=0.40
        )

        # Step 2: Use top three vector-retrieved chunks.
        evidence = retrieved[:3]

        print("Retrieved chunks:", len(retrieved))
        print("Evidence chunks used:", len(evidence))

        # Step 3: Evidence verification
        verification = verify_evidence(
            query,
            evidence
        )

        decision = verification["decision"]

        if decision == "ERROR":
            predicted = None
            result = "ERROR"

        else:
            predicted = decision == "SUPPORTED"
            result = (
                "PASS"
                if expected == predicted
                else "FAIL"
            )

        print("PREDICTED:", decision)
        print("REASON:", verification["reason"])
        print("RESULT:", result)

        results.append({
            "query": query,
            "expected": expected,
            "predicted": predicted,
            "decision": decision
        })

    # Step 4: Calculate metrics
    metrics = calculate_metrics(results)

    print("\n" + "=" * 65)
    print("FINAL EVALUATION RESULTS")
    print("=" * 65)

    print("Total tests:", len(results))
    print("Valid predictions:", metrics["valid"])
    print("Verification errors:", metrics["errors"])

    print("\nCONFUSION MATRIX COUNTS")
    print("-" * 40)
    print("True Positives  (TP):", metrics["TP"])
    print("True Negatives  (TN):", metrics["TN"])
    print("False Positives (FP):", metrics["FP"])
    print("False Negatives (FN):", metrics["FN"])

    print("\nCLASSIFICATION METRICS")
    print("-" * 40)
    print(f"Accuracy : {metrics['accuracy']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall   : {metrics['recall']:.4f}")
    print(f"F1-score : {metrics['f1']:.4f}")

    print("\n" + "=" * 65)
    print("EVALUATION COMPLETED")
    print("=" * 65)


if __name__ == "__main__":
    run_evaluation()
