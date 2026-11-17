
from src.retrieval import search_documents
from src.keyword_retrieval import search_keywords


TEST_CASES = [
    {
        "question": "Which indexing library and approximation were used to build the Wikipedia passage index?",
        "expected_page": 4
    },
    {
        "question": "What are the two main components of the RAG architecture?",
        "expected_page": 2
    },
    {
        "question": "How many Wikipedia passages were used in the December 2018 dataset?",
        "expected_page": 4
    },
    {
        "question": "Which neural retriever was used to initialize the document encoder?",
        "expected_page": 3
    },
    {
        "question": "What is the difference between RAG-Sequence and RAG-Token?",
        "expected_page": 2
    }
]

FILENAME = "rag_research_paper.pdf"
K = 60
TOP_K = 5


for test in TEST_CASES:

    query = test["question"]
    expected_page = test["expected_page"]

    vector_results = search_documents(
        query,
        n_results=10,
        max_distance=0.40,
        filename=FILENAME
    )

    keyword_results = search_keywords(
        query,
        n_results=10,
        filename=FILENAME
    )

    rrf_scores = {}
    documents = {}

    for ranking in (vector_results, keyword_results):

        for rank, item in enumerate(ranking, start=1):

            metadata = item["metadata"]

            document_id = (
                metadata["filename"],
                metadata["page"],
                metadata["chunk"]
            )

            rrf_scores[document_id] = (
                rrf_scores.get(document_id, 0)
                + 1 / (K + rank)
            )

            documents[document_id] = item

    ranked_results = sorted(
        rrf_scores.items(),
        key=lambda item: item[1],
        reverse=True
    )

    top_results = [
        documents[document_id]
        for document_id, score in ranked_results[:TOP_K]
    ]

    found_rank = next(
        (
            rank
            for rank, item in enumerate(top_results, start=1)
            if item["metadata"]["page"] == expected_page
        ),
        None
    )

    print("\n" + "=" * 65)
    print("QUESTION:", query)
    print("EXPECTED PAGE:", expected_page)
    print("EXPECTED PAGE RANK:", found_rank)

    for rank, item in enumerate(top_results, start=1):
        print(
            f"Rank {rank}: "
            f"Page {item['metadata']['page']} "
            f"| RRF {ranked_results[rank - 1][1]:.6f}"
        )

    if found_rank is not None:
        print("RESULT: EXPECTED PAGE RETRIEVED")
    else:
        print("RESULT: EXPECTED PAGE NOT IN TOP 5")
