# analyse_reviews.py

import re
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from load_data import load_data

# Load your reviews DataFrame
df = load_data("reviews")

# 1) Load a small SST-2–fine-tuned DistilBERT model once
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased-finetuned-sst-2-english")
model     = AutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased-finetuned-sst-2-english")
clf       = pipeline("sentiment-analysis", model=model, tokenizer=tokenizer)

def bert_sentiment(text: str) -> float:
    """
    Returns a sentiment score in [-1,+1] using the BERT pipeline.
    """
    out = clf(text[:512])[0]
    return out["score"] if out["label"] == "POSITIVE" else -out["score"]

def get_sentiment(text: str) -> float:
    """
    Splits a review into sentences, scores each with BERT,
    and returns the average. Falls back to VADER if needed.
    """
    sents = re.split(r'(?<=[.!?])\s+', text.strip())
    scores = [bert_sentiment(s) for s in sents if s]
    if not scores:
        return SentimentIntensityAnalyzer().polarity_scores(text)["compound"]
    return sum(scores) / len(scores)

if __name__ == "__main__":
    df["sentiment"] = df["text"].apply(get_sentiment)

    # Group by week and print metrics
    weekly_summary = df.groupby("week").agg(
        avg_sentiment=("sentiment", "mean"),
        avg_rating=("rating", "mean"),
        reviews_count=("text", "count")
    ).reset_index()

    print("Weekly Reviews Summary")
    for _, row in weekly_summary.iterrows():
        week = row["week"]
        sentiment = row["avg_sentiment"]
        rating = row["avg_rating"]
        count = row["reviews_count"]

        if sentiment >  0.3:
            mood = "Positive"
        elif sentiment < -0.3:
            mood = "Negative"
        else:
            mood = "Mixed"

        print(f"Week: {week}")
        print(f" - Avg Sentiment Score: {sentiment:.2f} ({mood})")
        print(f" - Avg Rating: {rating:.1f}")
        print(f" - Reviews Count: {count}\n")
