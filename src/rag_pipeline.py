from dataclasses import dataclass

@dataclass
class Document:
    id: str
    content: str

@dataclass
class RetrievalResult:
    query: str
    documents: list[Document]
    scores: list[float]

#Creating a document class
documents = [
    Document(
        id="doc1",
        content="Python is a programming language commonly used for AI and data science."
    ),
    Document(
        id="doc2",
        content="Azure AI Search is a cloud search service used to retrieve relevant information."
    ),
    Document(
        id="doc3",
        content="RAG combines information retrieval with language model generation."
    ),
]

#Retriver method
def retrieve(query: str):
    query_words = query.lower().split()

    results = []
    scores = []

    for document in documents:
        content = document.content.lower()

        matched_words = sum(
            word in content
            for word in query_words
        )

        if matched_words > 0:
            results.append(document)
            scores.append(matched_words / len(query_words))

    ranked = sorted(
        zip(results, scores),
        key=lambda item: item[1],
        reverse=True,
    )

    results = [item[0] for item in ranked]
    scores = [item[1] for item in ranked]

    return RetrievalResult(
        query=query,
        documents=results,
        scores=scores,
    )

def debug_retrieval(
    result: RetrievalResult,
    expected_document_ids: list[str] | None = None,
):
    best_document = None
    if result.documents:
        best_index = result.scores.index(max(result.scores))
        best_document = result.documents[best_index].id

    best_score = max(result.scores) if result.scores else 0.0
    retrieval_correct = (
            expected_document_ids is not None
            and best_document in expected_document_ids
    )
    recall = None

    if expected_document_ids:
        retrieved_ids = {document.id for document in result.documents}
        expected_ids = set(expected_document_ids)

        recall = len(retrieved_ids & expected_ids) / len(expected_ids)
    if not result.documents:
        status = "no_results"
    elif expected_document_ids is not None and not retrieval_correct:
        status = "incorrect"
    elif recall is not None and recall < 1.0:
        status = "weak"
    elif max(result.scores) < 1.0:
        status = "weak"
    else:
        status = "ok"

    if status == "no_results":
        message = "No relevant documents found."
    elif status == "incorrect":
        message = (
            f"Expected one of {expected_document_ids}, "
            f"but retrieved '{best_document}'."
        )
    elif status == "weak":
        message = "Documents found, but retrieval confidence is weak."
    else:
        message = "Relevant documents found."

    return {
        "query": result.query,
        "documents_found": len(result.documents),
        "status": status,
        "message": message,
        "best_document": best_document,
        "expected_documents": expected_document_ids,
        "retrieval_correct": retrieval_correct,
        "best_score": best_score,
        "recall": recall,
        "recall_percentage": recall * 100 if recall is not None else None,
        "documents": [
            {
                "id": document.id,
                "score": score,
            }
            for document, score in zip(result.documents, result.scores)
        ],
    }