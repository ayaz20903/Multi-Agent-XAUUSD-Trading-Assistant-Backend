from fastapi import APIRouter, HTTPException

from gold_ai.api.models import ChatRequest, ChatResponse, HealthResponse
from gold_ai.agents.orchestrator import graph

router = APIRouter()


@router.get("/health", response_model=HealthResponse)
def health():
    return HealthResponse()


@router.post("/api/chat", response_model=ChatResponse)
def chat(request: ChatRequest):
    try:
        result = graph.invoke(
            {
                "messages": [("user", request.message)],
                "agents": [],
            }
        )

        response_text = result["messages"][-1].content
        agents_used = result.get("agents", [])

        return ChatResponse(response=response_text, agents=agents_used)

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail="An internal error occurred while processing your request.",
        ) from exc
