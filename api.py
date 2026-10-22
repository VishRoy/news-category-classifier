import joblib
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict, Field
from db import init_db, log_prediction, get_prediction, save_feedback

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


class FeedbackRequest(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True)

    prediction_id: int
    correct_label: str


@app.post('/feedback')
def feedback(req: FeedbackRequest):
    if req.correct_label not in model.classes_.tolist():
        raise HTTPException(
            status_code=422,
            detail=f"Unknown category. Use one of: {model.classes_.tolist()}",
        )

    predicted = get_prediction(req.prediction_id)
    if predicted is None:
        raise HTTPException(status_code=404, detail="Prediction not found")

    is_correct = req.correct_label == predicted
    save_feedback(req.prediction_id, is_correct, req.correct_label)
    return {"status": "saved", "is_correct": is_correct}
