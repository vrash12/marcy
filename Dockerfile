# Python slim works well with sklearn wheels
FROM python:3.11-slim

# System deps (glibc locales optional)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy your app
COPY . .

# For Cloud Run: listen on $PORT; gunicorn is production-grade
ENV PYTHONUNBUFFERED=1
ENV PORT=8080
CMD exec gunicorn --bind :$PORT --workers 2 --threads 4 --timeout 120 app:app
