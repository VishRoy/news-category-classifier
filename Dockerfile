FROM python:3.13.12-slim

WORKDIR /app

COPY requirements-api.txt .
RUN pip install --no-cache-dir -r requirements-api.txt

RUN useradd --create-home appuser && chown appuser:appuser /app

COPY --chown=appuser:appuser api.py db.py model.joblib ./

USER appuser

EXPOSE 8000

CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]
