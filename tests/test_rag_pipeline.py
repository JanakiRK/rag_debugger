from src.rag_pipeline import retrieve
from src.rag_pipeline import debug_retrieval

def test_retrieve_python():
    result = retrieve("Python")

    assert result.query == "Python"
    assert len(result.documents) == 1
    assert result.documents[0].id == "doc1"
    assert result.scores == [1.0]

def test_retrieve_multiple_words():
    result = retrieve("AI data")

    assert len(result.documents) == 2
    assert result.documents[0].id == "doc1"
    assert result.documents[1].id == "doc2"
    assert result.scores == [1.0, 0.5]

def test_debug_retrieval():
    result = retrieve("AI data")
    debug = debug_retrieval(result)

    assert debug["query"] == "AI data"
    assert debug["documents_found"] == 2
    assert debug["documents"][0]["id"] == "doc1"
    assert debug["documents"][0]["score"] == 1.0

def test_retrieve_no_results():
    result = retrieve("JavaScript")

    assert len(result.documents) == 0
    assert result.scores == []

def test_debug_retrieval_no_results():
    result = retrieve("JavaScript")
    debug = debug_retrieval(result)

    assert debug["status"] == "no_results"
    assert debug["message"] == "No relevant documents found."

def test_debug_retrieval_weak():
    result = retrieve("Azure programming")

    debug = debug_retrieval(result)

    assert debug["status"] == "weak"
    assert debug["message"] == "Documents found, but retrieval confidence is weak."

def test_debug_retrieval_best_document():
    result = retrieve("AI data")
    debug = debug_retrieval(result)

    assert debug["best_document"] == "doc1"

def test_debug_retrieval_best_score():
    result = retrieve("AI data")
    debug = debug_retrieval(result)

    assert debug["best_score"] == 1.0

def test_debug_retrieval_expected_document():
    result = retrieve("RAG")

    debug = debug_retrieval(result, "doc3")

    assert debug["expected_document"] == "doc3"

def test_debug_retrieval_correct_document():
    result = retrieve("RAG")

    debug = debug_retrieval(result, "doc3")

    assert debug["retrieval_correct"] is True

def test_debug_retrieval_wrong_document():
    result = retrieve("RAG")

    debug = debug_retrieval(result, "doc1")

    assert debug["retrieval_correct"] is False