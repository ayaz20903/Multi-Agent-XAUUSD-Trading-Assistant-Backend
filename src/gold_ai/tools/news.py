import re

from tavily import TavilyClient
from langchain_core.tools import tool

from gold_ai.config import TAVILY_API_KEY


def clean_news_content(content: str, max_chars: int = 300) -> str:
    """Remove common news-page boilerplate and return concise article text."""
    if not content:
        return ""

    lines = [
        re.sub(r"\s+", " ", line).strip()
        for line in content.splitlines()
    ]

    lines = [line for line in lines if line]

    boilerplate_patterns = [
        r"^advertisement$",
        r"^latest news$",
        r"^home$",
        r"^subscribe$",
        r"^download app$",
        r"^sign in$",
        r"^login$",
        r"^menu$",
        r"^share$",
        r"^follow us$",
        r"^read more$",
    ]

    meaningful_lines = []

    for line in lines:
        if any(re.search(pattern, line, re.IGNORECASE) for pattern in boilerplate_patterns):
            continue

        meaningful_lines.append(line)

    cleaned = " ".join(meaningful_lines)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()

    return cleaned[:max_chars]


@tool
def search_gold_news(query: str):
    """Search the latest news related to gold, XAUUSD, and financial markets."""
    try:
        client = TavilyClient(api_key=TAVILY_API_KEY)

        response = client.search(
            query=query,
            topic="news",
            time_range="day",
            max_results=5,
        )

        news = []

        for result in response["results"]:
            news.append(
                {
                    "title": result["title"],
                    "url": result["url"],
                    "content": clean_news_content(result.get("content", "")),
                    "published_date": result.get("published_date"),
                }
            )

        if not news:
            return {
                "status": "success",
                "results": [],
            }

        return {
            "status": "success",
            "results": news,
        }

    except Exception:
        return {
            "status": "error",
            "error": "Unable to retrieve the latest gold news.",
        }


if __name__ == "__main__":
    results = search_gold_news.invoke(
        {"query": "XAUUSD gold price news"}
    )
    print(results)
