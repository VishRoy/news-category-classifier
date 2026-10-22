import joblib
from fastapi import FastAPI
from pydantic import BaseModel, ConfigDict, Field
from db import init_db, log_prediction

app = FastAPI(title='News Headline Classifier')
model = joblib.load('model.joblib')

MODEL_VERSION = "v1-tfidf-logreg-balanced"
init_db()


class PredictRequest(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True)
    headline: str = Field(min_length=1)


class PredictResponse(BaseModel):
    prediction_id: int
    category: str
    confidence: float


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": model is not None}


@app.post('/predict', response_model=PredictResponse)
def predict(req: PredictRequest):
    probs = model.predict_proba([req.headline])[0]
    best = probs.argmax()
    category = model.classes_[best]
    confidence = float(probs[best])

    prediction_id = log_prediction(
        req.headline, category, confidence, MODEL_VERSION)
    return PredictResponse(prediction_id=prediction_id, category=category, confidence=confidence)
