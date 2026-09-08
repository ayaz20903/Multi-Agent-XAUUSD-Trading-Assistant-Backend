from unittest.mock import patch, MagicMock

import pytest
from fastapi.testclient import TestClient

from gold_ai.api.app import app


client = TestClient(app)


def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"


@patch("gold_ai.api.routes.graph")
def test_chat_success(mock_graph):
    mock_graph.invoke.return_value = {
        "messages": [MagicMock(content="Gold is currently in a WAIT signal.")],
        "agents": ["technical", "market_context"],
    }

    response = client.post("/api/chat", json={"message": "What is the current signal?"})
    assert response.status_code == 200
    data = response.json()
    assert data["response"] == "Gold is currently in a WAIT signal."
    assert data["agents"] == ["technical", "market_context"]
    mock_graph.invoke.assert_called_once_with(
        {
            "messages": [("user", "What is the current signal?")],
            "agents": [],
        }
    )


def test_chat_missing_message():
    response = client.post("/api/chat", json={})
    assert response.status_code == 422


def test_chat_empty_message():
    response = client.post("/api/chat", json={"message": ""})
    assert response.status_code == 422


def test_chat_wrong_type():
    response = client.post("/api/chat", json={"message": 123})
    assert response.status_code == 422


@patch("gold_ai.api.routes.graph")
def test_chat_graph_error(mock_graph):
    mock_graph.invoke.side_effect = RuntimeError("LLM connection failed")

    response = client.post("/api/chat", json={"message": "test"})
    assert response.status_code == 500
    data = response.json()
    assert data["detail"] == "An internal error occurred while processing your request."


@patch("gold_ai.api.routes.graph")
def test_chat_returns_agents(mock_graph):
    mock_graph.invoke.return_value = {
        "messages": [MagicMock(content="Risk amount is $100.")],
        "agents": ["risk"],
    }

    response = client.post("/api/chat", json={"message": "Calculate risk for $10k at 1%."})
    assert response.status_code == 200
    data = response.json()
    assert data["agents"] == ["risk"]
