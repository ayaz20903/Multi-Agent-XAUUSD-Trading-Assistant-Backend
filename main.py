from gold_ai.agents.orchestrator import graph
from langchain_core.messages import AIMessage, ToolMessage


print("🤖 Gold AI Assistant")
print("Type 'exit' or 'quit' to stop.")

while True:
    query = input("\nYou: ")

    if query.lower() in ["exit", "quit"]:
        print("Goodbye!")
        break

    print("\nAI: ", end="", flush=True)

    active_tools = set()

    for message, metadata in graph.stream(
        {
            "messages": [
                ("user", query)
            ]
        },
        stream_mode="messages",
    ):

        # -----------------------------
        # AI MESSAGE
        # -----------------------------
        if isinstance(message, AIMessage):

            # Detect tool calls
            for tool_call in message.tool_calls:
                tool_name = tool_call["name"]

                if tool_name not in active_tools:
                    active_tools.add(tool_name)
                    print(
                        f"\n🔧 Calling {tool_name}...",
                        flush=True
                    )

            # Stream normal AI response
            if message.content:
                print(
                    message.content,
                    end="",
                    flush=True
                )

        # -----------------------------
        # TOOL MESSAGE
        # -----------------------------
        elif isinstance(message, ToolMessage):

            print(
                f"\n✅ {message.name} completed",
                flush=True
            )

    print()