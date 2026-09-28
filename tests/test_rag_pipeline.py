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