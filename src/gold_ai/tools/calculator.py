from langchain_core.tools import tool


@tool
def calculate_risk_reward(
    entry_price: float,
    stop_loss: float,
    take_profit: float,
):
    """Calculate the risk, reward, and risk-reward ratio for a trade."""

    risk = abs(entry_price - stop_loss)
    reward = abs(take_profit - entry_price)

    if risk == 0:
        return {
            "status": "error",
            "error": "Entry price and stop loss cannot be the same.",
        }

    risk_reward_ratio = reward / risk

    return {
        "status": "success",
        "entry_price": entry_price,
        "stop_loss": stop_loss,
        "take_profit": take_profit,
        "risk": round(risk, 3),
        "reward": round(reward, 3),
        "risk_reward_ratio": round(risk_reward_ratio, 2),
        "risk_to_reward": f"1:{round(risk_reward_ratio, 2)}",
    }

@tool
def calculate_risk_amount(
    account_balance: float,
    risk_percentage: float,
):
    """Calculate the monetary amount to risk based on account balance and risk percentage."""
    if account_balance <= 0:
        return {
            "status": "error",
            "error": "Account balance must be greater than zero.",
        }

    if risk_percentage <= 0:
        return {
            "status": "error",
            "error": "Risk percentage must be greater than zero.",
        }

    risk_amount = account_balance * (risk_percentage / 100)

    return {
        "status": "success",
        "account_balance": account_balance,
        "risk_percentage": risk_percentage,
        "risk_amount": round(risk_amount, 2),
    }

@tool
def calculate_position_size(
    account_balance: float,
    risk_percentage: float,
    stop_loss_distance: float,
    value_per_point: float,
):
    """Calculate position size based on account risk and stop-loss distance."""

    if account_balance <= 0:
        return {
            "status": "error",
            "error": "Account balance must be greater than zero.",
        }

    if risk_percentage <= 0:
        return {
            "status": "error",
            "error": "Risk percentage must be greater than zero.",
        }

    if stop_loss_distance <= 0:
        return {
            "status": "error",
            "error": "Stop-loss distance must be greater than zero.",
        }

    if value_per_point <= 0:
        return {
            "status": "error",
            "error": "Value per point must be greater than zero.",
        }

    risk_amount = account_balance * (risk_percentage / 100)

    position_size = risk_amount / (
        stop_loss_distance * value_per_point
    )

    return {
        "status": "success",
        "account_balance": account_balance,
        "risk_percentage": risk_percentage,
        "risk_amount": round(risk_amount, 2),
        "stop_loss_distance": stop_loss_distance,
        "value_per_point": value_per_point,
        "position_size": round(position_size, 4),
    }


if __name__ == "__main__":
    result = calculate_position_size.invoke({
        "account_balance": 10000,
        "risk_percentage": 1,
        "stop_loss_distance": 10,
        "value_per_point": 100,
    })

    print(result)