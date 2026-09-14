import json
from datetime import datetime, timezone
from urllib.request import urlopen
from urllib.error import URLError, HTTPError

from langchain_core.tools import tool


@tool
def get_gold_price():
    """Get the current XAUUSD gold price and its data freshness."""

    url = "https://api.gold-api.com/price/XAU"

    try:
        with urlopen(url, timeout=10) as response:
            data = json.load(response)

        price = data["price"]

        timestamp = data.get("updatedAt")

        if timestamp:
            as_of = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
            now = datetime.now(timezone.utc)
            age_seconds = (now - as_of).total_seconds()
            timestamp_iso = as_of.isoformat()
        else:
            age_seconds = None
            timestamp_iso = None

        return {
            "status": "success",
            "symbol": "XAUUSD",
            "price": price,
            "timestamp": timestamp_iso,
            "age_seconds": round(age_seconds, 2) if age_seconds is not None else None,
            "data_status": "fresh",
            "source": "Gold API",
        }

    except HTTPError as e:
        return {
            "status": "error",
            "error": f"Gold price API returned HTTP {e.code}."
        }

    except URLError:
        return {
            "status": "error",
            "error": "Unable to connect to the gold price API."
        }

    except (KeyError, TypeError, ValueError):
        return {
            "status": "error",
            "error": "Gold price API returned an unexpected response."
        }

    except Exception:
        return {
            "status": "error",
            "error": "An unexpected error occurred while retrieving gold price."
        }


if __name__ == "__main__":
    print(get_gold_price.invoke({}))