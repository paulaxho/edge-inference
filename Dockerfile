# Use stable Python version
ARG PYTHON_VERSION=3.10
FROM python:${PYTHON_VERSION}-slim as base

# Install system dependencies required for XGBoost (e.g., libgomp1)
RUN apt-get update && apt-get install -y \
    libgomp1 \
    gcc \
    python3-dev \
    libc-dev \
    build-essential \
    && rm -rf /var/lib/apt/lists/*


# Prevents Python from writing .pyc files
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
# Add these lines to your Dockerfile
ENV PYTHONFAULTHANDLER=1

# Set working directory
WORKDIR /app
RUN mkdir -p /app/results

# Create non-root user
ARG UID=10001
RUN adduser \
    --disabled-password \
    --gecos "" \
    --home "/nonexistent" \
    --shell "/sbin/nologin" \
    --no-create-home \
    --uid "${UID}" \
    appuser

# Copy in the requirements file and install dependencies
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

# Switch to non-root user
USER appuser

# Copy your application code
# - app/ directory (contains config.py, endpoints.py, logger.py, main.py)
# - src/ directory (contains inference.py, preprocess.py)
# - model/ directory (contains xgboost_model.json)
# - templates/ directory (for FastAPI Jinja2 templates)
# - style/ directory (if you need CSS or other static files)
COPY app/ /app/app/
COPY src/ /app/src/
COPY model/xgboost_model.json /app/xgboost_model.json
COPY templates/ /app/templates/
COPY style/ /app/style/

# (Optional) Copy data directory if you want data accessible inside the container
# COPY data/ /app/data/

# Expose the correct port (FastAPI default in your setup is 8000)
EXPOSE 8000

# Start the FastAPI server
# Here we reference "app.main:app", which means:
#  - 'app' is the package/folder
#  - 'main' is the module/file (main.py)
#  - 'app' is the FastAPI instance defined in main.py
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--proxy-headers", "--log-level", "debug"]