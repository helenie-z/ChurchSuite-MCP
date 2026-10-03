FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY server.py .

# Run as a non-root user
RUN useradd --create-home appuser
USER appuser

ENV HOST=0.0.0.0 \
    PORT=8000
EXPOSE 8000

# Streamable HTTP endpoint: http://<host>:8000/mcp
CMD ["python", "server.py"]
