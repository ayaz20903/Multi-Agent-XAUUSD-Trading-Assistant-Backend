import requests
from langchain_core.tools import tool


@tool
def get_gold_candles(timeframe: str, limit: int = 100):
    """Get recent completed XAUUSD OHLC candles for a specific timeframe."""

    allowed_timeframes = ["5m", "15m", "1h", "4h"]

    if timeframe not in allowed_timeframes:
        return {
            "status": "error",
            "error": f"Invalid timeframe. Use one of: {allowed_timeframes}"
        }

    if not 1 <= limit <= 1000:
        return {
            "status": "error",
            "error": "Limit must be between 1 and 1000."
        }

    try:
        response = requests.get(
            "https://biquote.io/api/XAUUSD/ohlc",
            params={
                "interval": timeframe,
                "limit": limit + 1,
            },
            headers={
                "User-Agent": "Mozilla/5.0",
                "Accept": "application/json",
            },
            timeout=10,
        )

        response.raise_for_status()

        data = response.json()

        candles = [
            {
                "timestamp": candle["openTime"],
                "open": candle["open"],
                "high": candle["high"],
                "low": candle["low"],
                "close": candle["close"],
                "tick_volume": candle["tickVolume"],
            }
            for candle in data["bars"]
            if not candle["isOpen"]
        ]

        candles = candles[:limit]

        return {
            "status": "success",
            "symbol": "XAUUSD",
            "timeframe": timeframe,
            "count": len(candles),
            "candles": candles,
        }

    except requests.exceptions.Timeout:
        return {
            "status": "error",
            "error": "Market data request timed out."
        }

    except requests.exceptions.RequestException:
        return {
            "status": "error",
            "error": "Unable to retrieve XAUUSD market data."
        }

    except (KeyError, TypeError, ValueError):
        return {
            "status": "error",
            "error": "Market data API returned an unexpected response."
        }

    except Exception:
        return {
            "status": "error",
            "error": "An unexpected error occurred while retrieving market data."
        }


if __name__ == "__main__":
    result = get_gold_candles.invoke({
        "timeframe": "5m",
        "limit": 5,
    })

    print(result)