import logging
import re
from datetime import datetime, timedelta

from fastapi import APIRouter, HTTPException

logger = logging.getLogger(__name__)

from gold_ai.api.models import (
    ChatRequest,
    ChatResponse,
    HealthResponse,
    MarketAnalysisResponse,
    PriceZone,
    StrategyDataResponse,
    TechnicalAnalysisData,
    MarketContextData,
)
from gold_ai.agents.orchestrator import graph
from gold_ai.strategy2 import (
    calculate_session_range,
    calculate_extension_levels,
    build_zones,
    detect_price_zone,
    find_recent_swing,
    detect_breakout,
    determine_trade_signal,
)
from gold_ai.technical_analysis import detect_trend
from gold_ai.tools.market_data import get_gold_candles
from gold_ai.tools.gold_price import get_gold_price
from gold_ai.tools.news import search_gold_news
from gold_ai.tools.economic_calendar import get_economic_calendar

router = APIRouter()

# Major XAUUSD-moving economic events
MAJOR_ECONOMIC_EVENTS = [
    "cpi",
    "nfp",
    "non-farm",
    "nonfarm",
    "fomc",
    "fed",
    "federal reserve",
    "interest rate",
    "rate decision",
    "pce",
    "gdp",
    "retail sales",
    "unemployment",
    "jobless",
    "payroll",
    "inflation",
    "consumer price",
    "producer price",
    "ppi",
]


def summarize_news(title: str, content: str) -> str:
    """
    Summarize a news headline into a short phrase explaining
    the main factor affecting gold prices.
    """
    if not title:
        return "—"

    title_lower = title.lower()

    # Pattern: Gold rises/falls on/after/due to [factor]
    patterns = [
        r"gold\s+(?:rises?|jumps?|climbs?|gains?|surges?|rallies?|advances?|hits?|reaches?)\s+(?:on|after|due|because|amid|as)\s+(.+?)(?:\s*[,;.]|$)",
        r"gold\s+(?:falls?|drops?|dips?|declines?|slips?|retreats?|pulled|pressured)\s+(?:on|after|due|because|amid|as)\s+(.+?)(?:\s*[,;.]|$)",
        r"xauusd\s+(?:rises?|jumps?|climbs?|gains?|surges?|rallies?|advances?|hits?|reaches?)\s+(?:on|after|due|because|amid|as)\s+(.+?)(?:\s*[,;.]|$)",
        r"xauusd\s+(?:falls?|drops?|dips?|declines?|slips?|retreats?|pulled|pressured)\s+(?:on|after|due|because|amid|as)\s+(.+?)(?:\s*[,;.]|$)",
    ]

    for pattern in patterns:
        match = re.search(pattern, title_lower)
        if match:
            factor = match.group(1).strip()
            # Clean up the factor
            factor = re.sub(r"\s+", " ", factor)
            if len(factor) > 40:
                factor = factor[:40].rsplit(" ", 1)[0]
            # Capitalize first letter
            return factor[0].upper() + factor[1:]

    # Check for common gold-related keywords
    if any(word in title_lower for word in ["dollar", "usd", "greenback"]):
        if any(word in title_lower for word in ["weak", "fall", "drop", "decline", "slide"]):
            return "Gold supported by weaker USD"
        elif any(word in title_lower for word in ["strong", "rise", "gain", "climb", "rally"]):
            return "Gold pressured by stronger USD"

    if any(word in title_lower for word in ["geopolitical", "war", "conflict", "tension", "crisis"]):
        return "Gold rises on safe-haven demand"

    if any(word in title_lower for word in ["rate", "fed", "fomc", "interest"]):
        if any(word in title_lower for word in ["cut", "lower", "easing", "dovish"]):
            return "Gold supported by rate cut expectations"
        elif any(word in title_lower for word in ["hike", "raise", "hawkish", "tighten"]):
            return "Gold pressured by rate hike expectations"

    if any(word in title_lower for word in ["inflation", "cpi", "pce"]):
        if any(word in title_lower for word in ["high", "rise", "jump", "surge"]):
            return "Gold gains on inflation concerns"
        elif any(word in title_lower for word in ["low", "fall", "drop", "decline"]):
            return "Gold weakens as inflation cools"

    if any(word in title_lower for word in ["yield", "treasury", "bond"]):
        if any(word in title_lower for word in ["rise", "jump", "surge", "climb"]):
            return "Gold pressured by rising yields"
        elif any(word in title_lower for word in ["fall", "drop", "decline", "slide"]):
            return "Gold supported by falling yields"

    # Fallback: use first part of title if it's short enough
    if len(title) <= 50:
        return title

    # Truncate long titles
    return title[:47] + "..."


def filter_major_economic_events(events: list) -> str:
    """
    Filter economic events to only show major XAUUSD-moving events.
    Returns a concise string of major events happening today.
    """
    if not events:
        return "—"

    major_events = []

    for event in events:
        event_name = event.get("name", "").lower()
        event_time = event.get("time", "")

        # Check if this is a major event
        is_major = any(keyword in event_name for keyword in MAJOR_ECONOMIC_EVENTS)

        if is_major:
            # Format the event name nicely
            name = event.get("name", "")
            # Truncate long names
            if len(name) > 30:
                name = name[:27] + "..."
            major_events.append(name)

    if not major_events:
        return "—"

    # Limit to 2 most important events
    if len(major_events) > 2:
        major_events = major_events[:2]

    return " + ".join(major_events)


@router.get("/health", response_model=HealthResponse)
def health():
    return HealthResponse()


@router.get("/api/strategy", response_model=StrategyDataResponse)
def get_strategy():
    try:
        candles_data = get_gold_candles.invoke({"timeframe": "5m", "limit": 300})

        if candles_data["status"] == "error":
            return StrategyDataResponse(
                status="error",
                reason=candles_data["error"],
            )

        candles = candles_data["candles"]

        session_range = calculate_session_range(candles)

        if session_range["status"] != "OK":
            return StrategyDataResponse(
                status="unavailable",
                reason=session_range.get("reason", "Session data unavailable."),
            )

        levels = calculate_extension_levels(session_range)

        if levels["status"] != "OK":
            return StrategyDataResponse(
                status="error",
                reason=levels.get("reason", "Could not calculate extension levels."),
            )

        zones = build_zones(levels)

        if zones["status"] != "OK":
            return StrategyDataResponse(
                status="error",
                reason=zones.get("reason", "Could not build trading zones."),
            )

        price_data = get_gold_price.invoke({})

        if price_data["status"] == "error":
            current_price = None
            signal_result = {"signal": "WAIT", "reason": "Price data unavailable."}
        else:
            current_price = price_data["price"]
            price_zone = detect_price_zone(current_price, zones)
            recent_candles = candles[:50]
            swing = find_recent_swing(recent_candles, lookback=10)
            breakout = detect_breakout(recent_candles, swing)
            signal_result = determine_trade_signal(price_zone, breakout)

        upside = levels["upside"]
        downside = levels["downside"]

        buy_zones = zones["buy_zones"]
        sell_zones = zones["sell_zones"]

        return StrategyDataResponse(
            session_high=session_range["session_high"],
            session_low=session_range["session_low"],
            range_size=session_range["range_size"],
            candle_count=session_range["candle_count"],
            buy_level_1=PriceZone(
                start=round(buy_zones["minor"]["low"], 3),
                end=round(buy_zones["minor"]["high"], 3),
            ),
            buy_level_2=PriceZone(
                start=round(buy_zones["major"]["low"], 3),
                end=round(buy_zones["major"]["high"], 3),
            ),
            sell_level_1=PriceZone(
                start=round(sell_zones["minor"]["low"], 3),
                end=round(sell_zones["minor"]["high"], 3),
            ),
            sell_level_2=PriceZone(
                start=round(sell_zones["major"]["low"], 3),
                end=round(sell_zones["major"]["high"], 3),
            ),
            signal=signal_result["signal"],
            status="ok",
        )

    except Exception as exc:
        logger.exception("Strategy endpoint failed")
        raise HTTPException(
            status_code=500,
            detail="An internal error occurred while fetching strategy data.",
        ) from exc


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
        logger.exception("Chat endpoint failed")
        raise HTTPException(
            status_code=500,
            detail="An internal error occurred while processing your request.",
        ) from exc


@router.get("/api/market-analysis", response_model=MarketAnalysisResponse)
def get_market_analysis():
    try:
        technical = TechnicalAnalysisData()
        market_context = MarketContextData()

        # --- Technical Analysis ---
        # Fetch 1h candles for trend detection
        candles_result = get_gold_candles.invoke({"timeframe": "1h", "limit": 50})

        if candles_result["status"] == "success":
            candles = candles_result["candles"]

            # Detect trend
            trend_result = detect_trend(candles, lookback=20)
            if trend_result["trend"] != "UNKNOWN":
                trend = trend_result["trend"].capitalize()
                technical.trend = trend

                # Derive structure from trend
                if trend == "Bullish":
                    technical.structure = "Higher highs / higher lows"
                    technical.liquidity = "Buy-side above recent high"
                elif trend == "Bearish":
                    technical.structure = "Lower highs / lower lows"
                    technical.liquidity = "Sell-side below recent low"
                else:
                    technical.structure = "Range-bound"
                    technical.liquidity = "Equal highs / equal lows"

        # Get signal from Strategy 2
        try:
            strategy_candles = get_gold_candles.invoke({"timeframe": "5m", "limit": 300})
            if strategy_candles["status"] == "success":
                session_range = calculate_session_range(strategy_candles["candles"])
                if session_range["status"] == "OK":
                    levels = calculate_extension_levels(session_range)
                    if levels["status"] == "OK":
                        zones = build_zones(levels)
                        if zones["status"] == "OK":
                            price_data = get_gold_price.invoke({})
                            if price_data["status"] == "success":
                                price_zone = detect_price_zone(price_data["price"], zones)
                                swing = find_recent_swing(strategy_candles["candles"][:50], lookback=10)
                                breakout = detect_breakout(strategy_candles["candles"][:50], swing)
                                signal_result = determine_trade_signal(price_zone, breakout)
                                technical.signal = signal_result["signal"]
        except Exception:
            logger.debug("Strategy signal retrieval failed", exc_info=True)
        # Fetch news and summarize
        try:
            news_result = search_gold_news.invoke({"query": "XAUUSD gold price"})
            if news_result["status"] == "success" and news_result["results"]:
                latest_news = news_result["results"][0]
                title = latest_news.get("title", "")
                content = latest_news.get("content", "")
                market_context.news = summarize_news(title, content)
        except Exception:
            logger.debug("News retrieval failed", exc_info=True)

        # Fetch economic events and filter for major events only
        try:
            today = datetime.now().strftime("%Y-%m-%d")
            calendar_result = get_economic_calendar.invoke({"date": today})
            if calendar_result["status"] == "success" and calendar_result["events"]:
                market_context.economic_events = filter_major_economic_events(
                    calendar_result["events"]
                )
        except Exception:
            logger.debug("Economic calendar retrieval failed", exc_info=True)

        return MarketAnalysisResponse(
            technical=technical,
            market_context=market_context,
            status="ok",
        )

    except Exception as exc:
        logger.exception("Market analysis endpoint failed")
        raise HTTPException(
            status_code=500,
            detail="An internal error occurred while fetching market analysis.",
        ) from exc
