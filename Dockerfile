FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .

RUN pip install --upgrade pip

# ✅ PyTorch 2.6+ required (security fix for CVE-2025-32434)
RUN pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cpu

RUN pip install --no-cache-dir -r requirements.txt

RUN python -m nltk.downloader punkt punkt_tab stopwords

COPY . .

EXPOSE 8000

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
