from transformers import AutoTokenizer, AutoModelForSequenceClassification #type: ignore
import torch #type: ignore

model_path = "model"
tokenizer_path = "tokenizer"

tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
model = AutoModelForSequenceClassification.from_pretrained(model_path)
model.eval()

label_map = {0: "negative 😡", 1: "neutral 😊", 2: "positive 😍"}

def predict(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        logits = model(**inputs).logits
    pred = torch.argmax(logits).item()
    return label_map[pred]

while True:
    user_input = input("Enter text: ")
    print("Sentiment:", predict(user_input))