# Manufacturing QC Agent Dockerfile
# ==================================
FROM python:3.11-slim

# Install system dependencies
# git: required for installing point9_platform
# libgl1, libglib2.0-0: required for opencv/yolo
RUN apt-get update && apt-get install -y \
    git \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Create a non-root user (Hugging Face Spaces requirement)
RUN useradd -m -u 1000 user

# Copy requirements file
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
# --chown=user ensures the non-root user owns the files
COPY --chown=user . .

# Create directory for artifacts if needed (and set permissions)
RUN mkdir -p outputs && chown user:user outputs

# Switch to non-root user
USER user

# Set environment variables
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH \
    PYTHONUNBUFFERED=1

# Expose the standard Hugging Face Spaces port
EXPOSE 7860

# Start the application
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "7860"]
