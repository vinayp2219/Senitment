from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

# --- Sentiment model ---
sentiment_model_name = "vinayp2219/sentiment-model-v2"
sentiment_tokenizer = AutoTokenizer.from_pretrained(sentiment_model_name)
sentiment_model = AutoModelForSequenceClassification.from_pretrained(sentiment_model_name)
sentiment_model.eval()

SENTIMENT_MAP = {
    0: ("negative", "😞"),
    1: ("neutral",  "😐"),
    2: ("positive", "😊"),
}

# --- GoEmotions model (28 emotions, no pipeline) ---
emotion_model_name = "SamLowe/roberta-base-go_emotions"
emotion_tokenizer = AutoTokenizer.from_pretrained(emotion_model_name)
emotion_model = AutoModelForSequenceClassification.from_pretrained(emotion_model_name)
emotion_model.eval()

EMOTION_EMOJIS = {
    "admiration":     "👏",
    "amusement":      "😄",
    "anger":          "😡",
    "annoyance":      "😤",
    "approval":       "👍",
    "caring":         "🤗",
    "confusion":      "😕",
    "curiosity":      "🧐",
    "desire":         "😍",
    "disappointment": "😞",
    "disapproval":    "👎",
    "disgust":        "🤢",
    "embarrassment":  "😳",
    "excitement":     "🤩",
    "fear":           "😨",
    "gratitude":      "🙏",
    "grief":          "😭",
    "joy":            "😄",
    "love":           "❤️",
    "nervousness":    "😬",
    "optimism":       "🌟",
    "pride":          "😤",
    "realization":    "💡",
    "relief":         "😮‍💨",
    "remorse":        "😔",
    "sadness":        "😢",
    "surprise":       "😲",
    "neutral":        "😑",
}


def predict(text):
    # Sentiment
    inputs = sentiment_tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        logits = sentiment_model(**inputs).logits
    sentiment_id = torch.argmax(logits).item()
    sentiment_label, sentiment_emoji = SENTIMENT_MAP[sentiment_id]

    # Emotion (GoEmotions — 28 classes, no pipeline)
    inputs = emotion_tokenizer(text[:512], return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        logits = emotion_model(**inputs).logits
    emotion_id = torch.argmax(logits).item()
    emotion_label = emotion_model.config.id2label[emotion_id].lower()
    emotion_emoji = EMOTION_EMOJIS.get(emotion_label, "🤔")

    return sentiment_label, sentiment_emoji, emotion_label, emotion_emoji


while True:
    user_input = input("\nEnter text (or 'quit' to exit): ").strip()
    if user_input.lower() == "quit":
        break
    if not user_input:
        continue

    sentiment, s_emoji, emotion, e_emoji = predict(user_input)
    print(f"  Sentiment : {sentiment.capitalize()} {s_emoji}")
    print(f"  Emotion   : {emotion.capitalize()} {e_emoji}")