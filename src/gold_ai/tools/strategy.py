from langchain_core.tools import tool

from gold_ai.retriever import retrieve_documents


_vector_store = None


def _get_vector_store():
    global _vector_store
    if _vector_store is None:
        from gold_ai.embeddings import create_embeddings
        from gold_ai.vector_store import load_vector_store

        embeddings = create_embeddings()
        _vector_store = load_vector_store(embeddings)
    return _vector_store


@tool
def search_strategy(query: str):
    """Search my XAUUSD trading strategy and return the most relevant information."""

    vector_store = _get_vector_store()

    results = retrieve_documents(
        vector_store,
        query,
    )

    if not results:
        return "No relevant information found in the trading strategy."

    best_document, best_score = results[0]

    return best_document.page_content


if __name__ == "__main__":
    result = search_strategy.invoke({
        "query": "What is the minimum risk-reward ratio?"
    })

    print(result)