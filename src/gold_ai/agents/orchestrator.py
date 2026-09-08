from typing import Annotated

from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from typing_extensions import TypedDict

from gold_ai.agents.supervisor import route_query
from gold_ai.agents.technical_agent import graph as technical_graph
from gold_ai.agents.market_context_agent import graph as market_context_graph
from gold_ai.agents.risk_agent import graph as risk_graph
from gold_ai.agents.final_agent import final_agent


class AgentState(TypedDict):
    messages: Annotated[list, add_messages]
    agents: list[str]


def supervisor(state: AgentState):
    query = state["messages"][-1].content

    decision = route_query(query)

    print(f"\nSupervisor selected: {decision.agents}")

    return {
        "agents": decision.agents
    }

def run_agents(state: AgentState):
    user_query = state["messages"][-1].content
    results = []

    for agent in state["agents"]:
        print(f"Running agent: {agent}")

        if agent == "technical":
            task = (
                f"Analyze the current XAUUSD technical situation for this user request:\n"
                f"{user_query}"
            )
            result = technical_graph.invoke({
                "messages": [("user", task)]
            })

        elif agent == "market_context":
            task = (
                f"Retrieve the relevant current gold/XAUUSD market context "
                f"for this user request. Do not provide a trading signal:\n"
                f"{user_query}"
            )
            result = market_context_graph.invoke({
                "messages": [("user", task)]
            })

        elif agent == "risk":
            task = (
                f"Provide risk-management information relevant to this user request. "
                f"Only perform a calculation if the required numbers are provided. "
                f"Otherwise report that the required risk information is unavailable:\n"
                f"{user_query}"
            )
            result = risk_graph.invoke({
                "messages": [("user", task)]
            })

        else:
            continue

        results.append(result["messages"][-1])

    return {"messages": results}



def final(state: AgentState):
    """
    Send all specialist results to the final decision agent.
    """

    return final_agent(state)


builder = StateGraph(AgentState)

builder.add_node("supervisor", supervisor)
builder.add_node("run_agents", run_agents)
builder.add_node("final", final)

builder.add_edge(START, "supervisor")
builder.add_edge("supervisor", "run_agents")
builder.add_edge("run_agents", "final")
builder.add_edge("final", END)

graph = builder.compile()


if __name__ == "__main__":

    test_queries = [
        "Should I take a trade on XAUUSD right now?",
        # "What is the current Strategy 2 signal?",
        # "What is the latest gold news?",
        # "Calculate the risk reward for entry 4400, stop loss 4390 and take profit 4420.",
    ]

    for query in test_queries:

        print("\n" + "=" * 70)
        print(f"USER: {query}")
        print("=" * 70)

        result = graph.invoke({
            "messages": [
                ("user", query)
            ],
            "agents": [],
        })

        print("\nFINAL RESPONSE:")
        print(result["messages"][-1].content)