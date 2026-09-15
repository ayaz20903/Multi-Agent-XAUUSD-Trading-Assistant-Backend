import re
from typing import Literal

from langgraph.graph import StateGraph, MessagesState, START
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_core.messages import SystemMessage, AIMessage
from pydantic import BaseModel

from gold_ai.rag import create_llm
from gold_ai.strategy2 import run_strategy2
from gold_ai.tools.gold_price import get_gold_price


llm = create_llm()

tools = [run_strategy2]

# LLM responsible only for deciding when to call tools
llm_with_tools = llm.bind_tools(tools)


class PriceZone(BaseModel):
    price: float
    status: str
    zone_type: str | None = None
    zone: str | None = None
    zone_low: float | None = None
    zone_high: float | None = None


class Breakout(BaseModel):
    broken_level: float | None = None
    close: float | None = None
    direction: str | None = None
    status: str
    timestamp: str | None = None


class TechnicalResult(BaseModel):
    signal: Literal["BUY", "SELL", "WAIT"]
    current_price: float | None
    price_zone: PriceZone | None
    breakout: Breakout | None
    reason: str


# Separate LLM responsible only for structured output
structured_llm = llm.with_structured_output(TechnicalResult)


_PRICE_QUERY_RE = re.compile(
    r"\b(current|live|latest|recent)\b.*\b(price|quote)\b"
    r"|\bprice\b.*\b(of\s+)?(gold|xau(?:usd)?)\b"
    r"|\b(gold|xau(?:usd)?)\b.*\bprice\b",
    re.IGNORECASE,
)


def _is_price_query(text: str) -> bool:
    return bool(_PRICE_QUERY_RE.search(text))


SYSTEM_PROMPT = """
You are the Technical Analysis Agent.

Use run_strategy2 whenever the user asks about:
- the current Strategy 2 signal
- Strategy 2 zones
- breakouts
- BUY / SELL / WAIT

IMPORTANT:
- Use the tool when current Strategy 2 information is required.
- Treat the tool result as the source of truth.
- Do not invent or modify numerical values.
- Do not perform calculations yourself.
"""


def chatbot(state: MessagesState):

    user_text = state["messages"][-1].content

    if _is_price_query(user_text):
        price_data = get_gold_price.invoke({})
        response = structured_llm.invoke(
            [
                SystemMessage(content=(
                    "You are formatting the result of a Technical Analysis Agent. "
                    "The user asked for the current gold price. "
                    "Convert the provided price data into the TechnicalResult structure. "
                    "The data contains a 'price' field — use its value as current_price. "
                    "Set signal to WAIT, price_zone and breakout to null. "
                    "Preserve all numerical values exactly."
                )),
                ("user", str(price_data)),
            ]
        )
        return {"messages": [AIMessage(content=response.model_dump_json())]}

    response = llm_with_tools.invoke(
        [
            SystemMessage(content=SYSTEM_PROMPT),
            *state["messages"],
        ]
    )

    return {
        "messages": [response]
    }


def format_result(state: MessagesState):

    tool_result = state["messages"][-1].content

    response = structured_llm.invoke(
        [
            SystemMessage(content="""
You are formatting the result of a Technical Analysis Agent.

Convert the provided Strategy 2 result into the TechnicalResult structure.

IMPORTANT:
- Preserve all numerical values exactly.
- Do not calculate anything.
- Do not invent missing information.
- If a value is unavailable, use null.
- Preserve BUY, SELL, or WAIT exactly.
"""),
            ("user", tool_result),
        ]
    )

    return {
        "messages": [
            AIMessage(content=response.model_dump_json())
        ]
    }


builder = StateGraph(MessagesState)

builder.add_node("chatbot", chatbot)
builder.add_node("tools", ToolNode(tools))
builder.add_node("format_result", format_result)

builder.add_edge(START, "chatbot")

builder.add_conditional_edges(
    "chatbot",
    tools_condition
)

builder.add_edge("tools", "format_result")

builder.add_edge("format_result", "__end__")

graph = builder.compile()


if __name__ == "__main__":

    result = graph.invoke({
        "messages": [
            ("user", "What is the current Strategy 2 signal?")
        ]
    })

    print(result["messages"][-1].content)