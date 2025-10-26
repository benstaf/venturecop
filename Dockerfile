# Use lightweight Python base
FROM python:3.11-slim

# Prevent interactive prompts and enable efficient pip installs
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Set working directory
WORKDIR /app

# Copy dependencies first for Docker caching
COPY requirements.txt .

# Install system dependencies and Python packages
RUN apt-get update && apt-get install -y --no-install-recommends build-essential \
    && pip install -r requirements.txt \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# Copy the rest of the app
COPY . .

# Expose Streamlit’s default port
EXPOSE 8501

# Run Streamlit app (update to your file)
CMD ["streamlit", "run", "app_pdf.py", "--server.port=8501", "--server.address=0.0.0.0"]
