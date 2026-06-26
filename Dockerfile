FROM python:3.13-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app/src

WORKDIR /app

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY src ./src
COPY scripts ./scripts
COPY data ./data
COPY artifacts/.retrain_loop_state.json ./artifacts/.retrain_loop_state.json

CMD ["python", "scripts/run_retrain_loop.py", "--once"]
