import os

import joblib
import pytest
from fastapi.testclient import TestClient
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline


@pytest.fixture(scope="session")
def client(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("ci")

    texts = [
        "Lakers win the championship", "Team scores late goal",
        "Senate passes new bill", "President signs law",
        "Apple launches new iPhone", "Startup releases AI app",
    ]
    labels = ["SPORTS", "SPORTS", "POLITICS",
              "POLITICS", "SCIENCE & TECH", "SCIENCE & TECH"]
    model = Pipeline([("tfidf", TfidfVectorizer()),
                     ("clf", LogisticRegression())])
    model.fit(texts, labels)

    model_path = tmp / "model.joblib"
    joblib.dump(model, model_path)
    os.environ["MODEL_PATH"] = str(model_path)

    import db
    db.DB_PATH = str(tmp / "test.db")

    import api
    return TestClient(api.app)


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["model_loaded"] is True


def test_predict_returns_valid_response(client):
    r = client.post("/predict", json={"headline": "Lakers win again"})
    assert r.status_code == 200
    body = r.json()
    assert body["category"] in {"SPORTS", "POLITICS", "SCIENCE & TECH"}
    assert 0 <= body["confidence"] <= 1
    assert isinstance(body["prediction_id"], int)


@pytest.mark.parametrize("payload", [
    {"headline": ""},
    {"headline": "   "},
    {"title": "Lakers win"},
])
def test_predict_rejects_bad_input(client, payload):
    r = client.post("/predict", json=payload)
    assert r.status_code == 422


def test_feedback_rejects_unknown_label(client):
    r = client.post(
        "/feedback", json={"prediction_id": 1, "correct_label": "SPORT"})
    assert r.status_code == 422
    assert "SPORTS" in r.json()["detail"]


def test_feedback_unknown_prediction_id(client):
    r = client.post(
        "/feedback", json={"prediction_id": 99999, "correct_label": "SPORTS"})
    assert r.status_code == 404


def test_predict_then_feedback(client):
    pred = client.post(
        "/predict", json={"headline": "Senate passes budget"}).json()
    wrong = "SPORTS" if pred["category"] != "SPORTS" else "POLITICS"

    r = client.post(
        "/feedback", json={"prediction_id": pred["prediction_id"], "correct_label": wrong})
    assert r.status_code == 200
    assert r.json()["is_correct"] is False

    r = client.post(
        "/feedback", json={"prediction_id": pred["prediction_id"], "correct_label": pred["category"]})
    assert r.json()["is_correct"] is True
