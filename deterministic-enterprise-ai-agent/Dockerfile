FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1
WORKDIR /app

RUN apt-get update && apt-get install -y build-essential && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
# We combine these to ensure the layer is fresh
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt && \
    pip list | grep langchain  # This will show us exactly what is installed

COPY . .

# Using 'python -m' ensures the correct sys.path is used
CMD ["python", "-m", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080"]