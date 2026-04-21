FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .

RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

RUN python -m nltk.downloader punkt stopwords

RUN pip install torch==2.4.0 --index-url https://download.pytorch.org/whl/cpu

COPY . .

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
