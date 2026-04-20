from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import FileResponse, JSONResponse, HTMLResponse
from fastapi.middleware.cors import CORSMiddleware

from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline

from wordcloud import WordCloud
import matplotlib.pyplot as plt

import torch
import os
import nltk

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.probability import FreqDist

nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('stopwords')



app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

UPLOAD_FOLDER = "uploads"
WC_FOLDER = "wordclouds"

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(WC_FOLDER, exist_ok=True)


# -----------------------
# SENTIMENT MODEL (your trained model)
# -----------------------
model_name = "vinayp2219/sentiment-model-v2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)


# -----------------------
# EMOTION MODEL (pre-trained, no training needed)
# Downloads ~300MB on first run, cached after that
# -----------------------

emotion_pipeline = pipeline(
    "text-classification",
    model="j-hartmann/emotion-english-distilroberta-base",
    top_k=1
)

EMOTION_EMOJIS = {
    "joy":      "😄",
    "anger":    "😡",
    "disgust":  "🤢",
    "fear":     "😨",
    "sadness":  "😢",
    "surprise": "😲",
    "neutral":  "😑",
}


# -----------------------
# FILE TEXT EXTRACTION
# -----------------------

def extract_file_text(file_bytes):
    try:
        return file_bytes.decode(errors="ignore")[:5000]
    except:
        return ""


# -----------------------
# WORDCLOUD
# -----------------------

def generate_wordcloud(text, path):
    wc = WordCloud(
        width=1200,
        height=600,
        background_color="white"
    ).generate(text)

    plt.figure(figsize=(14, 7))
    plt.imshow(wc, interpolation="bilinear")
    plt.axis("off")
    plt.savefig(path, bbox_inches="tight")
    plt.close()


# -----------------------
# NLTK TEXT SUMMARIZER
# -----------------------

def text_summarizer(text, num_sentences=3):
    sentences = sent_tokenize(text)

    if len(sentences) <= num_sentences:
        return text

    words = word_tokenize(text.lower())
    stop_words = set(stopwords.words("english"))
    filtered_words = [
        word for word in words
        if word.isalnum() and word not in stop_words
    ]

    fdist = FreqDist(filtered_words)

    sentence_scores = []
    for sentence in sentences:
        score = 0
        for word in word_tokenize(sentence.lower()):
            if word in fdist:
                score += fdist[word]
        sentence_scores.append(score)

    sentence_scores = list(enumerate(sentence_scores))
    sorted_sentences = sorted(sentence_scores, key=lambda x: x[1], reverse=True)
    top_sentences = sorted(sorted_sentences[:num_sentences], key=lambda x: x[0])

    return " ".join([sentences[i] for i, _ in top_sentences])


# -----------------------
# FRONTEND
# -----------------------

@app.get("/", response_class=HTMLResponse)
def serve_home():
    return FileResponse("templates/index.html")


# -----------------------
# WORDCLOUD ROUTE
# -----------------------

@app.get("/wordcloud/{name}")
def get_wordcloud(name: str):
    return FileResponse(os.path.join(WC_FOLDER, name))


# -----------------------
# MAIN API
# -----------------------

@app.post("/predict")
async def predict(
    text: str = Form(""),
    file: UploadFile | None = File(None)
):
    if file:
        content = await file.read()
        text = extract_file_text(content)

    text = text.strip()

    if not text:
        return JSONResponse({"error": "No text found"}, status_code=400)

    # --- Sentiment prediction (your trained model) ---
    encoded = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True
    )

    with torch.no_grad():
        logits = model(**encoded).logits

    probs = torch.softmax(logits, dim=1).numpy()[0]
    labels = ["negative", "neutral", "positive"]
    label = labels[int(probs.argmax())]

    # --- Emotion prediction (pre-trained model) ---
    # Truncate to 512 chars for the emotion model to avoid token limit issues
    emotion_result = emotion_pipeline(text[:512])[0]
    emotion = emotion_result[0]["label"].lower()
    emotion_emoji = EMOTION_EMOJIS.get(emotion, "🤔")

    # --- Summary ---
    summary = text_summarizer(text)

    # --- Word cloud ---
    wc_name = f"wc_{torch.randint(0, 999999, (1,)).item()}.png"
    wc_path = os.path.join(WC_FOLDER, wc_name)
    generate_wordcloud(text, wc_path)

    return {
        "label": label,
        "scores": {
            "negative": float(probs[0]),
            "neutral":  float(probs[1]),
            "positive": float(probs[2])
        },
        "emotion":       emotion,
        "emotion_emoji": emotion_emoji,
        "summary":       summary,
        "wordcloud_url": f"/wordcloud/{wc_name}"
    }
