from tavily import TavilyClient
from langchain_core.tools import tool

from gold_ai.config import TAVILY_API_KEY


@tool
def search_gold_news(query: str):
    """Search the latest news related to gold, XAUUSD, and financial markets."""

    try:
        client = TavilyClient(api_key=TAVILY_API_KEY)

        response = client.search(
            query=query,
            topic="news",
            time_range="day",
            max_results=5
        )

        news = []

        for result in response["results"]:
            news.append({
                "title": result["title"],
                "url": result["url"],
                "content": result["content"],
                "published_date": result.get("published_date")
            })

        if not news:
            return {
                "status": "success",
                "results": []
            }

        return {
            "status": "success",
            "results": news
        }

    except Exception:
        return {
            "status": "error",
            "error": "Unable to retrieve the latest gold news."
        }


if __name__ == "__main__":
    results = search_gold_news.invoke({
        "query": "XAUUSD gold price news"
    })

    print(results)