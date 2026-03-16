# Dockerfile for ChoreoAI API
FROM python:3.11-slim

# Create a non-privileged user
RUN useradd -m choreouser
WORKDIR /app

# Install system dependencies required for OpenCV and MediaPipe
RUN apt-get update && apt-get install -y \
    ffmpeg \
    libsm6 \
    libxext6 \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy project files
COPY . .
RUN chown -R choreouser:choreouser /app

# Install only production dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir . && \
    pip install --no-cache-dir fastapi pydantic uvicorn

USER choreouser
# Set python path
ENV PYTHONPATH=/app/src

# Expose API port
EXPOSE 8000

# Default command: run the FastAPI server
CMD ["uvicorn", "choreoai.api:app", "--host", "0.0.0.0", "--port", "8000"]
