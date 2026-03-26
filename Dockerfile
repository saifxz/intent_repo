# Stage 1: Builder
FROM python:3.11-slim AS builder

WORKDIR /app

# Install build essentials if any packages need compilation
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
# Install to a local folder to easily copy in the next stage
RUN pip install --user --no-cache-dir -r requirements.txt

# Stage 2: Runtime
FROM python:3.11-slim AS runtime

WORKDIR /app

COPY --from=builder /root/.local /root/.local
COPY --from=builder /app/requirements.txt .

ENV PATH=/root/.local/bin:$PATH

# --- ML Model Management --
COPY download_models.py .
RUN python download_models.py

COPY models/ ./models/
COPY inference.py main.py ./

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]