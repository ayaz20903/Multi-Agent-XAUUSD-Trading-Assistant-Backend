import ast
import json

from langgraph.graph import StateGraph, MessagesState, START
from langgraph.prebuilt import ToolNode
from langchain_core.messages import SystemMessage, HumanMessage

from gold_ai.rag import create_llm
from gold_ai.tools.news import search_gold_news
from gold_ai.tools.economic_calendar import get_economic_calendar

from pydantic import BaseModel


class NewsItem(BaseModel):
    title: str
    url: str
    content: str
    published_date: str | None = None


class EconomicEvent(BaseModel):
    name: str | None = None
    date: str | None = None
    time: str | None = None
    actual: str | float | int | None = None
    forecast: str | float | int | None = None
    previous: str | float | int | None = None


class MarketContextResult(BaseModel):
    news: list[NewsItem]
    economic_events: list[EconomicEvent]
    summary: str


llm = create_llm()

tools = [
    search_gold_news,
    get_economic_calendar,
]

llm_with_tools = llm.bind_tools(tools)


SYSTEM_PROMPT = """
You are the Market Context Agent for a Gold/XAUUSD trading assistant.

Your responsibility is ONLY market context.

You have access to:

- search_gold_news
- get_economic_calendar

Use search_gold_news when the user asks about:
- Latest gold news
- XAUUSD news
- Current market news
- Recent events affecting gold

Use get_economic_calendar when the user asks about:
- USD economic events
- Economic releases
- High-impact US events
- Economic calendar information

IMPORTANT RULES:

- Use the appropriate tool to retrieve current information.
- Treat tool results as the source of truth.
- Do not invent news or economic events.
- Do not invent market reactions or causal relationships.
- Do not provide technical analysis.
- Do not provide BUY, SELL, or WAIT signals.
- Do not modify information returned by the tools.
- Clearly distinguish factual information from interpretation.
- Keep the response concise and factual.

Your output will later be passed to another agent
that will synthesize the overall response.
"""


def market_context_agent(state: MessagesState):
    response = llm_with_tools.invoke(
        [
            SystemMessage(content=SYSTEM_PROMPT),
            *state["messages"],
        ]
    )

    return {
        "messages": [response]
    }


def route_after_agent(state: MessagesState):
    last_message = state["messages"][-1]

    has_tool_results = any(
        m.type == "tool" for m in state["messages"]
    )

    if last_message.tool_calls and not has_tool_results:
        return "tools"

    return "format_result"


def format_result(state: MessagesState):

    print("\nRAW TOOL RESULTS:")

    news = []
    economic_events = []

    for message in state["messages"]:

        if message.type != "tool":
            continue

        print(message.content)

        tool_name = message.name
        tool_result = message.content

        # ToolMessage content can be a string or a dictionary.
        if isinstance(tool_result, str):

            try:
                result_data = json.loads(tool_result)

            except json.JSONDecodeError:

                try:
                    result_data = ast.literal_eval(tool_result)

                except (ValueError, SyntaxError):
                    continue

        elif isinstance(tool_result, dict):

            result_data = tool_result

        else:
            continue

        if not isinstance(result_data, dict):
            continue

        if result_data.get("status") != "success":
            continue

        if tool_name == "search_gold_news":

            for item in result_data.get("results", []):

                news.append(
                    NewsItem(
                        title=item.get("title", ""),
                        url=item.get("url", ""),
                        content=item.get("content", ""),
                        published_date=item.get("published_date"),
                    )
                )

        elif tool_name == "get_economic_calendar":
            for item in result_data.get("events", []):
                event_date = item.get("date")
                event_time = item.get("time")

                # Some calendar API responses provide the date/time
                # together inside the time field.
                if event_time and not event_date:
                    event_time_str = str(event_time)

                    if "T" in event_time_str:
                        event_date = event_time_str.split("T")[0]
                    elif " " in event_time_str:
                        event_date = event_time_str.split(" ")[0]

                economic_events.append(
                    EconomicEvent(
                        name=item.get("name"),
                        date=event_date,
                        time=event_time,
                        actual=item.get("actual"),
                        forecast=item.get("forecast"),
                        previous=item.get("previous"),
                    )
                )

    seen_urls = set()
    unique_news = []
    for item in news:
        if item.url not in seen_urls:
            seen_urls.add(item.url)
            unique_news.append(item)

    result = MarketContextResult(
        news=unique_news,
        economic_events=economic_events,
        summary="Market context retrieved successfully.",
    )

    return {
        "messages": [
            HumanMessage(
                content=result.model_dump_json()
            )
        ]
    }


builder = StateGraph(MessagesState)

builder.add_node("market_context", market_context_agent)
builder.add_node("tools", ToolNode(tools))
builder.add_node("format_result", format_result)

builder.add_edge(START, "market_context")

builder.add_conditional_edges(
    "market_context",
    route_after_agent,
    {
        "tools": "tools",
        "format_result": "format_result",
    },
)

builder.add_edge("tools", "market_context")
builder.add_edge("format_result", "__end__")

graph = builder.compile()


if __name__ == "__main__":

    result = graph.invoke({
        "messages": [
            (
                "user",
                "What is the latest gold news and what high-impact US economic events are coming up?"
            )
        ]
    })

    print("\nFINAL RESULT:")
    print(result["messages"][-1].content)
