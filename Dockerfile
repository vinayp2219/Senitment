FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .

RUN pip install --upgrade pip

# Install PyTorch CPU FIRST (before requirements to avoid conflicts)
RUN pip install torch==2.4.0 --index-url https://download.pytorch.org/whl/cpu

# Install rest of dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Download ALL 3 NLTK packages your app.py needs
RUN python -m nltk.downloader punkt punkt_tab stopwords

COPY . .

EXPOSE 8000

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
