from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    message: str = Field(
        ..., min_length=1, description="The user message to send to the assistant"
    )


class ChatResponse(BaseModel):
    response: str = Field(..., description="The assistant's response")
    agents: list[str] = Field(
        default_factory=list,
        description="List of specialist agents that were invoked",
    )


class HealthResponse(BaseModel):
    status: str = "ok"
