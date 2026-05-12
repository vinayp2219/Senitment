from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import FileResponse, JSONResponse, HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from nltk.tokenize import sent_tokenize
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline

from wordcloud import WordCloud
import matplotlib.pyplot as plt

import torch
import os
import nltk
import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.probability import FreqDist

# nltk.download('punkt')
# nltk.download('punkt_tab')
# nltk.download('stopwords')



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
        return " ".join(sentences)  # clean join instead of raw text

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
    
    # Pick top N but cap at half the total sentences so it's always a reduction
    top_n = min(num_sentences, max(1, len(sentences) // 2))
    top_sentences = sorted(sorted_sentences[:top_n], key=lambda x: x[0])

    return " ".join([sentences[i] for i, _ in top_sentences])

def predict_single(text):
    encoded = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True
    )

    with torch.no_grad():
        logits = model(**encoded).logits

    probs = torch.softmax(logits, dim=1).numpy()[0]
    neg, neu, pos = probs

    if pos > neg and pos > neu:
        return "positive", probs
    elif neg > pos and neg > neu:
        return "negative", probs
    else:
        return "neutral", probs

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



# Split on contrast words like "but", "however"
    parts = re.split(r'\bbut\b|\bhowever\b|\bthough\b', text.lower())
    sentences = []

    for part in parts:
        sentences.extend(sent_tokenize(part))

    sentence_labels = []
    all_probs = []

    for s in sentences:
        lbl, probs = predict_single(s)
        sentence_labels.append(lbl)
        all_probs.append(probs)

# Average probabilities
    avg_probs = sum(all_probs) / len(all_probs)
    neg, neu, pos = avg_probs

# Final decision
    if "positive" in sentence_labels and "negative" in sentence_labels:
        label = "mixed"
    elif pos > neg and pos > neu:
        label = "positive"
    elif neg > pos and neg > neu:
        label = "negative"
    else:
        label = "neutral"

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
            "negative": round(float(avg_probs[0]) * 100, 2),
            "neutral":  round(float(avg_probs[1]) * 100, 2),
            "positive": round(float(avg_probs[2]) * 100, 2)
        },
        "emotion":       emotion,
        "emotion_emoji": emotion_emoji,
        "summary":       summary,
        "wordcloud_url": f"/wordcloud/{wc_name}"
    }
