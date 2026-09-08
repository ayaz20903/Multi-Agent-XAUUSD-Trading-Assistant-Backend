from langchain_core.tools import tool
from datetime import datetime
from zoneinfo import ZoneInfo
from gold_ai.tools.gold_price import get_gold_price
from gold_ai.tools.market_data import get_gold_candles

def calculate_session_range(
    candles,
    session_start="09:30",
    session_end="10:00",
):
    """
    Calculate the session high, session low, and range size
    for the 09:30–10:00 IST session.

    Candles are expected to contain:
        timestamp
        high
        low

    The session is interpreted in Asia/Kolkata timezone.
    """

    if not candles:
        return {
            "status": "UNKNOWN",
            "reason": "No candle data available."
        }

    ist = ZoneInfo("Asia/Kolkata")

    start_hour, start_minute = map(int, session_start.split(":"))
    end_hour, end_minute = map(int, session_end.split(":"))

    session_candles = []

    for candle in candles:
        timestamp = candle["timestamp"]

        # Handle ISO timestamps returned by the API
        if isinstance(timestamp, str):
            timestamp = datetime.fromisoformat(
                timestamp.replace("Z", "+00:00")
            )

        local_time = timestamp.astimezone(ist)

        start_time = local_time.replace(
            hour=start_hour,
            minute=start_minute,
            second=0,
            microsecond=0,
        )

        end_time = local_time.replace(
            hour=end_hour,
            minute=end_minute,
            second=0,
            microsecond=0,
        )

        if start_time <= local_time < end_time:
            session_candles.append(candle)

    if not session_candles:
        return {
            "status": "UNKNOWN",
            "reason": "No candles found for the session."
        }

    session_high = max(
        candle["high"]
        for candle in session_candles
    )

    session_low = min(
        candle["low"]
        for candle in session_candles
    )

    range_size = session_high - session_low

    return {
        "status": "OK",
        "session_high": round(session_high, 3),
        "session_low": round(session_low, 3),
        "range_size": round(range_size, 3),
        "candle_count": len(session_candles),
    }


def calculate_extension_levels(session_range):
    """
    Calculate Minor and Major upside/downside levels
    from the session range.
    """

    if session_range["status"] != "OK":
        return {
            "status": "UNKNOWN",
            "reason": "Session range is not available."
        }

    session_high = session_range["session_high"]
    session_low = session_range["session_low"]
    range_size = session_range["range_size"]

    return {
        "status": "OK",

        "upside": {
            "minor": {
                "level_1": round(session_high + 2.00 * range_size, 3),
                "level_2": round(session_high + 2.25 * range_size,3),
                "level_3": round(session_high + 2.50 * range_size,3),
            },
            "major": {
                "level_1": round(session_high + 4.00 * range_size,3),
                "level_2": round(session_high + 4.25 * range_size,3),
                "level_3": round(session_high + 4.50 * range_size,3),
            },
        },

        "downside": {
            "minor": {
                "level_1": round(session_low - 2.00 * range_size, 3),
                "level_2": round(session_low - 2.25 * range_size, 3),
                "level_3": round(session_low - 2.50 * range_size, 3),
            },
            "major": {
                "level_1": round(session_low - 4.00 * range_size, 3),
                "level_2": round(session_low - 4.25 * range_size, 3),
                "level_3": round(session_low - 4.50 * range_size, 3),
            },
        },
    }

def build_zones(levels):
    """
    Convert the extension levels into Minor and Major
    buy/sell zones.
    """

    if levels["status"] != "OK":
        return {
            "status": "UNKNOWN",
            "reason": "Extension levels are not available."
        }

    upside = levels["upside"]
    downside = levels["downside"]

    return {
        "status": "OK",

        "sell_zones": {
            "minor": {
                "low": upside["minor"]["level_1"],
                "high": upside["minor"]["level_3"],
            },
            "major": {
                "low": upside["major"]["level_1"],
                "high": upside["major"]["level_3"],
            },
        },

        "buy_zones": {
            "minor": {
                "low": downside["minor"]["level_3"],
                "high": downside["minor"]["level_1"],
            },
            "major": {
                "low": downside["major"]["level_3"],
                "high": downside["major"]["level_1"],
            },
        },
    }

def detect_price_zone(current_price, zones):
    """
    Determine whether the current price is inside
    a Minor/Major buy or sell zone.
    """

    if zones["status"] != "OK":
        return {
            "status": "UNKNOWN",
            "reason": "Zones are not available."
        }

    for zone_type, zone_group in [
        ("BUY", zones["buy_zones"]),
        ("SELL", zones["sell_zones"]),
    ]:
        for zone_name, zone in zone_group.items():
            if zone["low"] <= current_price <= zone["high"]:
                return {
                    "status": "IN_ZONE",
                    "zone_type": zone_type,
                    "zone": zone_name.upper(),
                   "price": round(current_price, 3),
                    "zone_low": zone["low"],
                    "zone_high": zone["high"],
                }

    return {
        "status": "NO_ZONE",
        "price": round(current_price, 3),
    }

def find_recent_swing(candles, lookback=10):
    """
    Find the most recent local swing high and swing low
    from completed 5M candles.

    Candles are ordered newest -> oldest.
    """

    if len(candles) < lookback:
        return {
            "status": "UNKNOWN",
            "reason": f"Need at least {lookback} candles."
        }

    recent = candles[:lookback]

    swing_high = None
    swing_low = None

    for i in range(1, len(recent) - 1):

        previous_candle = recent[i - 1]
        current_candle = recent[i]
        next_candle = recent[i + 1]

        if swing_high is None:
            if (
                current_candle["high"] > previous_candle["high"]
                and current_candle["high"] > next_candle["high"]
            ):
                swing_high = current_candle

        if swing_low is None:
            if (
                current_candle["low"] < previous_candle["low"]
                and current_candle["low"] < next_candle["low"]
            ):
                swing_low = current_candle

        if swing_high is not None and swing_low is not None:
            break

    if swing_high is None or swing_low is None:
        return {
            "status": "UNKNOWN",
            "reason": "Could not identify both a swing high and swing low."
        }

    return {
        "status": "OK",
        "swing_high": round(swing_high["high"], 3),
        "swing_low": round(swing_low["low"], 3),
        "high_timestamp": swing_high["timestamp"],
        "low_timestamp": swing_low["timestamp"],
    }

def detect_breakout(candles, swing):
    """
    Detect whether the latest completed 5M candle
    has broken the recent swing high or swing low.
    """

    if swing["status"] != "OK":
        return {
            "status": "UNKNOWN",
            "reason": "Swing levels are not available."
        }

    latest_candle = candles[0]

    swing_high = swing["swing_high"]
    swing_low = swing["swing_low"]

    if latest_candle["close"] > swing_high:
        return {
            "status": "BREAKOUT",
            "direction": "BULLISH",
            "close": latest_candle["close"],
            "broken_level": swing_high,
            "timestamp": latest_candle["timestamp"],
        }

    if latest_candle["close"] < swing_low:
        return {
            "status": "BREAKOUT",
            "direction": "BEARISH",
            "close": latest_candle["close"],
            "broken_level": swing_low,
            "timestamp": latest_candle["timestamp"],
        }

    return {
        "status": "NO_BREAKOUT",
        "close": latest_candle["close"],
        "swing_high": swing_high,
        "swing_low": swing_low,
        "timestamp": latest_candle["timestamp"],
    }

def determine_trade_signal(price_zone, breakout):
    """
    Combine price zone and 5M breakout confirmation
    to determine BUY, SELL, or WAIT.
    """

    if price_zone["status"] != "IN_ZONE":
        return {
            "signal": "WAIT",
            "reason": "Price is not inside a trading zone."
        }

    if breakout["status"] != "BREAKOUT":
        return {
            "signal": "WAIT",
            "reason": "Price is inside a zone, but there is no confirmed breakout."
        }

    zone_type = price_zone["zone_type"]
    breakout_direction = breakout["direction"]

    if zone_type == "BUY" and breakout_direction == "BULLISH":
        return {
            "signal": "BUY",
            "zone": price_zone["zone"],
            "reason": "Price is in a BUY zone and a bullish breakout is confirmed."
        }

    if zone_type == "SELL" and breakout_direction == "BEARISH":
        return {
            "signal": "SELL",
            "zone": price_zone["zone"],
            "reason": "Price is in a SELL zone and a bearish breakout is confirmed."
        }

    return {
        "signal": "WAIT",
        "reason": "Zone direction and breakout direction do not match."
    }

@tool
def run_strategy2():
    """
    Run the complete Strategy 2 analysis.

    Strategy 2 uses completed 5-minute candles.
    The timeframe cannot be changed.

    """

    # Get enough 5M candles to reliably include
    # the 09:30–10:00 session.
    candles_data = get_gold_candles.invoke({
    "timeframe": "5m",
    "limit": 300,
    })

    if candles_data["status"] == "error":
        return {
            "status": "error",
            "error": candles_data["error"],
            "stage": "market_data",
        }

    candles = candles_data["candles"]

    # 1. Session range
    session_range = calculate_session_range(candles)

    if session_range["status"] != "OK":
        return session_range

    # 2. Extension levels
    levels = calculate_extension_levels(session_range)

    # 3. Trading zones
    zones = build_zones(levels)

    # 4. Current price
    price_data = get_gold_price.invoke({})

    if price_data["status"] == "error":
        return {
            "status": "error",
            "error": price_data["error"],
            "stage": "gold_price",
        }

    current_price = price_data["price"]

    # 5. Current price zone
    price_zone = detect_price_zone(
        current_price,
        zones
    )

    # 6. Recent 5M market structure
    recent_candles = candles[:50]

    swing = find_recent_swing(
        recent_candles,
        lookback=10
    )

    # 7. Breakout
    breakout = detect_breakout(
        recent_candles,
        swing
    )

    # 8. Final signal
    signal = determine_trade_signal(
        price_zone,
        breakout
    )

    return {
        "current_price": round(current_price, 3),
        "session_range": session_range,
        "levels": levels,
        "zones": zones,
        "price_zone": price_zone,
        "swing": swing,
        "breakout": breakout,
        "signal": signal,
    }



if __name__ == "__main__":
    result = run_strategy2.invoke({})

    print("\nStrategy 2 Result:")
    print(result)