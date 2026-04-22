FROM python:3.10-slim

WORKDIR /app

# Install system dependencies needed for building packages
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

RUN pip install --upgrade pip

# Install PyTorch CPU-only (slim version, no CUDA = much smaller)
RUN pip install torch==2.6.0+cpu --index-url https://download.pytorch.org/whl/cpu \
    --no-cache-dir

# Install rest of requirements
RUN pip install --no-cache-dir -r requirements.txt

# Download NLTK data
RUN python -m nltk.downloader punkt punkt_tab stopwords

COPY . .

EXPOSE 8000

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
