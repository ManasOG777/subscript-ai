FROM python:3.11-slim

# Install ffmpeg and system deps
RUN apt-get update && apt-get install -y \
    ffmpeg \
    git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python deps first (cached layer)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy app
COPY app.py .
COPY templates/ templates/

# Create runtime dirs
RUN mkdir -p uploads outputs

EXPOSE 8080

CMD ["gunicorn", "app:app", \
     "--bind", "0.0.0.0:8080", \
     "--timeout", "600", \
     "--workers", "2", \
     "--threads", "4", \
     "--worker-class", "gthread"]
