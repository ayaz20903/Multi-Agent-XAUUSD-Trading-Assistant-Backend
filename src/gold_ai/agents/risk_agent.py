import ast

from typing import Literal
from langgraph.graph import StateGraph, MessagesState, START
from langgraph.prebuilt import ToolNode
from langchain_core.messages import SystemMessage, AIMessage
from pydantic import BaseModel
from gold_ai.rag import create_llm
from gold_ai.tools.calculator import (
    calculate_risk_amount,
    calculate_risk_reward,
    calculate_position_size,
)
from gold_ai.tools.strategy import search_strategy

llm = create_llm()

tools = [
    calculate_risk_amount,
    calculate_risk_reward,
    calculate_position_size,
    search_strategy,
]

llm_with_tools = llm.bind_tools(tools)


# ---------------------------------------------------------
# Structured output
# ---------------------------------------------------------

class RiskResult(BaseModel):
    status: Literal["success", "error"]
    result_type: Literal["calculation", "strategy_rule"]

    entry_price: float | None = None
    stop_loss: float | None = None
    take_profit: float | None = None

    risk: float | None = None
    reward: float | None = None
    risk_reward_ratio: float | None = None
    risk_to_reward: str | None = None

    account_balance: float | None = None
    risk_percentage: float | None = None
    risk_amount: float | None = None

    stop_loss_distance: float | None = None
    value_per_point: float | None = None
    position_size: float | None = None

    reason: str


# ---------------------------------------------------------
# Agent prompt
# ---------------------------------------------------------

SYSTEM_PROMPT = """
You are the Risk Agent for a Gold/XAUUSD trading assistant.

Your responsibility is ONLY risk management and trade calculations.

TOOLS:

1. calculate_risk_amount
Use this when the user provides:
- account balance
- risk percentage

and wants to know how much money they should risk.

Example:
Account = $10,000
Risk = 1%
→ calculate the risk amount using this tool.

2. calculate_risk_reward

Use this when the user asks to calculate:
- risk
- reward
- risk-reward ratio
- R:R

when entry price, stop loss, and take profit are provided.

3. calculate_position_size

Use this when the user wants to calculate position/lot size.

Required information:
- account balance
- risk percentage
- stop-loss distance
- value per point

IMPORTANT:
- Do NOT assume a value per point.
- Do NOT use a default value.
- XAUUSD contract specifications can vary by broker.
- If value per point is not provided, do not calculate position size.
- Report that the value per point is required.

4. search_strategy

Use this for ANY question asking about rules or information
contained in the user's trading strategy knowledge base.

Examples:
- "What is the minimum risk-reward ratio?"
- "What RR does my strategy require?"
- "What are my risk management rules?"
- "What does my strategy say about poor risk-reward?"

IMPORTANT TOOL SELECTION:

- If the user asks about a strategy rule, you MUST call
  search_strategy.

- If the user asks for a calculation, you MUST call the
  appropriate calculator tool.

- Do NOT answer strategy-rule questions from your general
  knowledge.

IMPORTANT:

- Treat tool results as the source of truth.
- Do not perform calculations yourself.
- Do not modify numerical values.
- Do not invent missing information.
- Do not provide technical analysis.
- Do not provide BUY, SELL, or WAIT signals.
"""


# ---------------------------------------------------------
# Risk Agent
# ---------------------------------------------------------

def risk_agent(state: MessagesState):
    response = llm_with_tools.invoke(
        [
            SystemMessage(content=SYSTEM_PROMPT),
            *state["messages"],
        ]
    )

    return {
        "messages": [response]
    }


# ---------------------------------------------------------
# Routing
# ---------------------------------------------------------

def route_after_agent(state: MessagesState):
    last_message = state["messages"][-1]

    if last_message.tool_calls:
        return "tools"

    return "format_result"


# ---------------------------------------------------------
# Format tool result
# ---------------------------------------------------------

def format_result(state: MessagesState):

    tool_messages = [
        message
        for message in state["messages"]
        if message.type == "tool"
    ]

    # No tool was called
    if not tool_messages:
        # Check if the LLM produced a text response instead of calling a tool
        ai_messages = [
            message
            for message in state["messages"]
            if message.type == "ai" and message.content
        ]

        if ai_messages:
            reason = ai_messages[-1].content
        else:
            reason = "No risk-management tool result was produced."

        result = RiskResult(
            status="error",
            result_type="calculation",
            reason=reason,
        )

        return {
            "messages": [
                AIMessage(
                    content=result.model_dump_json()
                )
            ]
        }

    # Use the latest tool result
    tool_message = tool_messages[-1]

    tool_name = tool_message.name
    tool_result = tool_message.content

    # -----------------------------------------------------
    # Convert tool result into Python dictionary
    # -----------------------------------------------------

    if isinstance(tool_result, str):

        try:
            result_data = ast.literal_eval(tool_result)

        except (ValueError, SyntaxError):

            result_data = {
                "raw_result": tool_result
            }

    elif isinstance(tool_result, dict):

        result_data = tool_result

    else:

        result_data = {
            "raw_result": str(tool_result)
        }

    # -----------------------------------------------------
    # Strategy knowledge result
    # -----------------------------------------------------

    if tool_name == "search_strategy":

        if "raw_result" in result_data:
            strategy_text = result_data["raw_result"]
        else:
            strategy_text = str(result_data)

        result = RiskResult(
            status="success",
            result_type="strategy_rule",
            reason=strategy_text,
        )

    # -------------------------------------------
    # Risk/reward calculation
    # -------------------------------------------

    elif tool_name == "calculate_risk_amount":
        if result_data.get("status") == "error":
            result = RiskResult(
                status="error",
                result_type="calculation",
                reason=result_data.get(
                    "error",
                    "Risk-amount calculation failed."
                ),
            )
        else:
            result = RiskResult(
                status="success",
                result_type="calculation",
                account_balance=result_data.get("account_balance"),
                risk_percentage=result_data.get("risk_percentage"),
                risk_amount=result_data.get("risk_amount"),
                reason="Risk amount calculation completed.",
            )

    elif tool_name == "calculate_risk_reward":

        if result_data.get("status") == "error":

            result = RiskResult(
                status="error",
                result_type="calculation",
                reason=result_data.get(
                    "error",
                    "Risk-reward calculation failed."
                ),
            )

        else:

            result = RiskResult(
                status="success",
                result_type="calculation",
                entry_price=result_data.get("entry_price"),
                stop_loss=result_data.get("stop_loss"),
                take_profit=result_data.get("take_profit"),
                risk=result_data.get("risk"),
                reward=result_data.get("reward"),
                risk_reward_ratio=result_data.get(
                    "risk_reward_ratio"
                ),
                risk_to_reward=result_data.get(
                    "risk_to_reward"
                ),

                reason="Risk-reward calculation completed.",
            )

    # -----------------------------------------------------
    # Position size calculation
    # -----------------------------------------------------

    elif tool_name == "calculate_position_size":

        if result_data.get("status") == "error":
            return {
                "messages": [
                    AIMessage(
                        content=RiskResult(
                            status="error",
                            result_type="calculation",
                            reason=result_data["error"],
                        ).model_dump_json()
                    )
                ]
            }

        return {
            "messages": [
                AIMessage(
                    content=RiskResult(
                        status="success",
                        result_type="calculation",
                        account_balance=result_data.get("account_balance"),
                        risk_percentage=result_data.get("risk_percentage"),
                        risk_amount=result_data.get("risk_amount"),
                        stop_loss_distance=result_data.get("stop_loss_distance"),
                        value_per_point=result_data.get("value_per_point"),
                        position_size=result_data.get("position_size"),
                        reason="Position size calculation completed.",
                    ).model_dump_json()
                )
            ]
        }

    # -----------------------------------------------------
    # Unknown tool
    # -----------------------------------------------------

    else:

        result = RiskResult(
            status="error",
            result_type="calculation",
            reason="Unknown risk-management tool result.",
        )

    return {
        "messages": [
            AIMessage(
                content=result.model_dump_json()
            )
        ]
    }


# ---------------------------------------------------------
# Build graph
# ---------------------------------------------------------

builder = StateGraph(MessagesState)

builder.add_node(
    "risk_agent",
    risk_agent
)

builder.add_node(
    "tools",
    ToolNode(tools)
)

builder.add_node(
    "format_result",
    format_result
)

builder.add_edge(
    START,
    "risk_agent"
)

builder.add_conditional_edges(
    "risk_agent",
    route_after_agent,
    {
        "tools": "tools",
        "format_result": "format_result",
    },
)

builder.add_edge(
    "tools",
    "risk_agent"
)

builder.add_edge(
    "format_result",
    "__end__"
)

graph = builder.compile()


# ---------------------------------------------------------
# Test
# ---------------------------------------------------------

if __name__ == "__main__":

    result = graph.invoke(
        {
            "messages": [
                (
                    "user",
                    "Calculate position size for a $10,000 account risking 1% with a 10-point stop loss."
                )
            ]
        }
    )

    print("\nFINAL RESULT:")
    print(result["messages"][-1].content)