FROM python:3.12-slim

# Set working directory
WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application files
COPY . .

# Expose port (if using SSE transport, FastMCP typically uses port 8000)
EXPOSE 8000

# Run the application
CMD ["python", "main.py"]

