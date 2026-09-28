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

def debug_retrieval(result: RetrievalResult):
    status = "ok" if result.documents else "no_results"

    return {
        "query": result.query,
        "documents_found": len(result.documents),
        "status": status,
        "documents": [
            {
                "id": document.id,
                "score": score,
            }
            for document, score in zip(result.documents, result.scores)
        ],
    }