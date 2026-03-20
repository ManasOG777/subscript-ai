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

# Pre-download Whisper model files into the image during build.
# This bakes tiny/base/small into the image so they load instantly at runtime
# with no HuggingFace downloads. Medium is also included; the ~2.2 GB extra
# image size is worth eliminating cold-start delays for all model sizes.
RUN python3 -c "\
from huggingface_hub import snapshot_download; \
repos = [ \
    'Systran/faster-whisper-tiny', \
    'Systran/faster-whisper-base', \
    'Systran/faster-whisper-small', \
    'Systran/faster-whisper-medium', \
]; \
[snapshot_download(repo_id=r) for r in repos]; \
print('All Whisper models cached.')"

# Copy app
COPY app.py .
COPY templates/ templates/

# Create runtime dirs
RUN mkdir -p uploads outputs

EXPOSE $PORT

CMD gunicorn app:app \
    --bind 0.0.0.0:${PORT:-8080} \
    --timeout 600 \
    --workers 1 \
    --threads 4 \
    --worker-class gthread
