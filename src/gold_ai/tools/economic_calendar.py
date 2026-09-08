from datetime import datetime, timedelta

import requests
from langchain_core.tools import tool


@tool
def get_economic_calendar(date: str):
    """Get high-impact U.S. economic events for a specific date."""

    try:
        start_date = datetime.strptime(date, "%Y-%m-%d")
        end_date = start_date + timedelta(days=1)

        params = {
            "from": date,
            "to": end_date.strftime("%Y-%m-%d"),
            "countries": "US",
            "importance": "high",
            "limit": 50,
        }

        response = requests.get(
            "https://biquote.io/api/calendar",
            params=params,
            headers={
                "User-Agent": "Mozilla/5.0",
                "Accept": "application/json",
            },
            timeout=10,
        )

        response.raise_for_status()

        return {
            "status": "success",
            "date": date,
            "timezone": "UTC",
            "events": response.json()
        }

    except ValueError:
        return {
            "status": "error",
            "error": "Invalid date format. Use YYYY-MM-DD."
        }

    except requests.exceptions.Timeout:
        return {
            "status": "error",
            "error": "Economic calendar request timed out."
        }

    except requests.exceptions.RequestException:
        return {
            "status": "error",
            "error": "Unable to retrieve the economic calendar."
        }

    except Exception:
        return {
            "status": "error",
            "error": "An unexpected error occurred while retrieving the economic calendar."
        }


if __name__ == "__main__":
    result = get_economic_calendar.invoke({
        "date": "2026-09-04"
    })

    print(result)