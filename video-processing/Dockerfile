FROM python:3.9-slim

# Environment variables
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PORT=5000

WORKDIR /app

# Install system dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends ffmpeg curl && \
    rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir --timeout 1000 --upgrade pip setuptools wheel
RUN pip install --no-cache-dir --timeout 1000 -r requirements.txt

# Copy application
COPY . .

# Create uploads folder
RUN mkdir -p /app/uploads && chmod -R 777 /app/uploads

# Expose Flask port
EXPOSE 5000

# Run Gunicorn
CMD ["gunicorn", "--bind", "0.0.0.0:5000", "--timeout", "1501", "app:app"]
