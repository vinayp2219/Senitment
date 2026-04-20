from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline  # type: ignore
import torch  # type: ignore

model_name = "vinayp2219/sentiment-model-v2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)
model.eval()

SENTIMENT_MAP = {
    0: ("negative", "😞"),
    1: ("neutral",  "😐"),
    2: ("positive", "😊"),
}

# --- Emotion model (pre-trained, downloads once ~300MB) ---
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


def predict(text):
    # Sentiment
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        logits = model(**inputs).logits
    sentiment_id = torch.argmax(logits).item()
    sentiment_label, sentiment_emoji = SENTIMENT_MAP[sentiment_id]

    # Emotion
    emotion_result = emotion_pipeline(text[:512])[0]
    emotion = emotion_result[0]["label"].lower()
    emotion_emoji = EMOTION_EMOJIS.get(emotion, "🤔")

    return sentiment_label, sentiment_emoji, emotion, emotion_emoji


while True:
    user_input = input("\nEnter text (or 'quit' to exit): ").strip()
    if user_input.lower() == "quit":
        break
    if not user_input:
        continue

    sentiment, s_emoji, emotion, e_emoji = predict(user_input)
    print(f"  Sentiment : {sentiment.capitalize()} {s_emoji}")
    print(f"  Emotion   : {emotion.capitalize()} {e_emoji}")
