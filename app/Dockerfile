FROM python:3.11.14-slim-trixie

# Do not write .pyc files, flush stdout/stderr immediately
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# Only copy dependency file first (for better caching)
COPY requirements.txt .

# System dependencies:
# - build-essential + image libs for pillow/matplotlib
# - docker.io for Docker CLI inside the container
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    zlib1g-dev \
    libjpeg-dev \
    libpng-dev \
    libfreetype6-dev \
    docker.io \
  && pip install --no-cache-dir -r requirements.txt \
  && apt-get clean \
  && rm -rf /var/lib/apt/lists/*

# Copy the rest of the app source code
COPY . .

# Default command: run the interactive agent
CMD ["python", "main_agent.py"]
