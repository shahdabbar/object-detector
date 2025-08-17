FROM python:3.10-slim

# Create the working directory
RUN set -ex && mkdir /app
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Copy the relevant directories
COPY requirements.txt .
COPY model/ ./model
COPY . ./

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip \
 && pip install --no-cache-dir -r requirements.txt

# Create output dir (where annotated images are saved)
RUN mkdir -p static/outputs

# Run the web server
EXPOSE 8000
ENV PYTHONPATH /app
CMD ["gunicorn", "--workers=1", "--threads=1", "--timeout=120", "--bind", "0.0.0.0:8000", "app:app"]
