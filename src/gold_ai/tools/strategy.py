from langchain_core.tools import tool

from gold_ai.embeddings import create_embeddings
from gold_ai.vector_store import load_vector_store
from gold_ai.retriever import retrieve_documents


embeddings = create_embeddings()
vector_store = load_vector_store(embeddings)


@tool
def search_strategy(query: str):
    """Search my XAUUSD trading strategy and return the most relevant information."""

    results = retrieve_documents(
        vector_store,
        query
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