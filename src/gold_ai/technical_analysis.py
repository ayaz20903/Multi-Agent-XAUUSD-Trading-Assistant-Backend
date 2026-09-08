def detect_trend(candles, lookback=20):
    """
    Detect trend using basic market structure.

    Bullish  = Higher High + Higher Low
    Bearish  = Lower High + Lower Low
    Otherwise = RANGING
    """

    if len(candles) < lookback:
        return {
            "trend": "UNKNOWN",
            "reason": f"Need at least {lookback} candles."
        }

    recent = candles[:lookback]

    # Split the candles into two periods:
    # older candles vs newer candles
    midpoint = len(recent) // 2

    newer = recent[:midpoint]
    older = recent[midpoint:]

    newer_high = max(c["high"] for c in newer)
    newer_low = min(c["low"] for c in newer)

    older_high = max(c["high"] for c in older)
    older_low = min(c["low"] for c in older)

    if newer_high > older_high and newer_low > older_low:
        trend = "BULLISH"

    elif newer_high < older_high and newer_low < older_low:
        trend = "BEARISH"

    else:
        trend = "RANGING"

    return {
        "trend": trend,
        "newer_high": newer_high,
        "newer_low": newer_low,
        "older_high": older_high,
        "older_low": older_low,
    }


def calculate_fibonacci_zone(candles, lookback=50):
    """
    Identify the dominant impulse over the recent 1H candles
    and calculate the 0.382 - 0.5 Fibonacci retracement zone.
    """

    if len(candles) < lookback:
        return {
            "status": "UNKNOWN",
            "reason": f"Need at least {lookback} candles."
        }

    recent = candles[:lookback]

    # Find the highest high and lowest low
    swing_high = max(recent, key=lambda candle: candle["high"])
    swing_low = min(recent, key=lambda candle: candle["low"])

    high_index = recent.index(swing_high)
    low_index = recent.index(swing_low)

    # Candles are newest -> oldest.
    #
    # If the low happened earlier and the high happened later,
    # the market made a bullish impulse: Low -> High.
    #
    # If the high happened earlier and the low happened later,
    # the market made a bearish impulse: High -> Low.

    direction = determine_impulse_direction(
        recent,
        swing_high,
        swing_low
    )

    if direction == "BULLISH":
        impulse_low = swing_low["low"]
        impulse_high = swing_high["high"]

        price_range = impulse_high - impulse_low

        fib_382 = impulse_high - (price_range * 0.382)
        fib_500 = impulse_high - (price_range * 0.5)

    elif direction == "BEARISH":
        impulse_high = swing_high["high"]
        impulse_low = swing_low["low"]

        price_range = impulse_high - impulse_low

        fib_382 = impulse_low + (price_range * 0.382)
        fib_500 = impulse_low + (price_range * 0.5)

    else:
        return {
            "status": "UNKNOWN",
            "reason": "Unable to determine impulse direction."
        }

    zone_high = max(fib_382, fib_500)
    zone_low = min(fib_382, fib_500)

    current_price = recent[0]["close"]

    in_zone = zone_low <= current_price <= zone_high

    return {
        "status": "OK",
        "direction": direction,
        "swing_high": impulse_high,
        "swing_low": impulse_low,
        "fib_0.382": fib_382,
        "fib_0.5": fib_500,
        "zone_high": zone_high,
        "zone_low": zone_low,
        "current_price": current_price,
        "in_zone": in_zone,
    }


def determine_impulse_direction(candles, swing_high, swing_low):
    """
    Determine whether the move between the selected swing high
    and swing low is bullish or bearish.

    Candles are ordered newest -> oldest.
    """

    high_index = candles.index(swing_high)
    low_index = candles.index(swing_low)

    if low_index > high_index:
        return "BULLISH"

    elif high_index > low_index:
        return "BEARISH"

    return "UNKNOWN"



if __name__ == "__main__":
    from gold_ai.tools.market_data import get_gold_candles

    result = get_gold_candles.invoke({
        "timeframe": "1h",
        "limit": 50,
    })

    analysis = calculate_fibonacci_zone(
        result["candles"]
    )

    print(analysis)