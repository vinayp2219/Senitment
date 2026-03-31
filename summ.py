import random
import nltk

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.probability import FreqDist

nltk.download("punkt")
nltk.download("stopwords")


def text_summarizer(text, num_sentences=3):

    sentences = sent_tokenize(text)

    words = word_tokenize(text.lower())

    stop_words = set(stopwords.words("english"))

    filtered_words = [
        word for word in words
        if word.casefold() not in stop_words
    ]

    fdist = FreqDist(filtered_words)

    sentence_scores = [
        sum(
            fdist[word]
            for word in word_tokenize(sentence.lower())
            if word in fdist
        )
        for sentence in sentences
    ]

    sentence_scores = list(enumerate(sentence_scores))

    sorted_sentences = sorted(sentence_scores, key=lambda x: x[1], reverse=True)

    random_sentences = random.sample(sorted_sentences, num_sentences)

    summary_sentences = sorted(random_sentences, key=lambda x: x[0])

    summary = " ".join([sentences[i] for i, _ in summary_sentences])

    return summary


# -------- FILE INPUT --------

file_path = input("Enter file path: ")

with open(file_path, "r", encoding="utf-8") as f:
    text = f.read()

summary = text_summarizer(text)

print("\n----- SUMMARY -----\n")
print(summary)