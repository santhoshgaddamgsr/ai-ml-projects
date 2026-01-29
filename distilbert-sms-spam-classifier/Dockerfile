FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY main.py .

# Use the shell form to allow $PORT substitution
CMD exec uvicorn main:app --host 0.0.0.0 --port ${PORT:-8080}