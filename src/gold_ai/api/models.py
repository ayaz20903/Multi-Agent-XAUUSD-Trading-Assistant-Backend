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


class StrategyDataResponse(BaseModel):
    session_high: float | None = Field(None, description="Session high price")
    session_low: float | None = Field(None, description="Session low price")
    range_size: float | None = Field(None, description="Session range size")
    candle_count: int | None = Field(None, description="Number of session candles")
    buy_level_1: float | None = Field(None, description="Buy zone level 1 (minor)")
    buy_level_2: float | None = Field(None, description="Buy zone level 2 (major)")
    sell_level_1: float | None = Field(None, description="Sell zone level 1 (minor)")
    sell_level_2: float | None = Field(None, description="Sell zone level 2 (major)")
    signal: str = Field("WAIT", description="Current trading signal")
    status: str = Field("ok", description="Data availability status")
    reason: str | None = Field(None, description="Reason if data unavailable")


class TechnicalAnalysisData(BaseModel):
    trend: str = Field("—", description="Market trend direction")
    structure: str = Field("—", description="Market structure description")
    liquidity: str = Field("—", description="Liquidity condition")
    signal: str = Field("—", description="Trading signal")


class MarketContextData(BaseModel):
    news: str = Field("—", description="Summarized gold market news")
    economic_events: str = Field("—", description="Major economic events relevant to XAUUSD today")


class MarketAnalysisResponse(BaseModel):
    technical: TechnicalAnalysisData = Field(
        default_factory=TechnicalAnalysisData,
        description="Technical analysis data",
    )
    market_context: MarketContextData = Field(
        default_factory=MarketContextData,
        description="Market context data",
    )
    status: str = Field("ok", description="Data availability status")
    reason: str | None = Field(None, description="Reason if data unavailable")
