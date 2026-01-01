FROM python:3.12-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Install PyTorch with CUDA from PyTorch index
# For CUDA 12.8, use cu128 index
# Note: For GPU support, you need nvidia-docker runtime in docker-compose.yml
RUN pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cu128

# Copy requirements first for better caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application files
COPY . .

# Expose port
EXPOSE 8082

# Run the application
CMD ["python", "main.py"]

