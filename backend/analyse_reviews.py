from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import re
from load_data import load_data

df = load_data("reviews")

# 2) load a small, SST-2-fine-tuned model
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased-finetuned-sst-2-english")
model     = AutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased-finetuned-sst-2-english")
clf       = pipeline("sentiment-analysis", model=model, tokenizer=tokenizer)

# 3) helper to get a [-1, +1] score (the sentiment)
def bert_sentiment(text):
    # truncate to 512 tokens to avoid warnings
    text = text if len(text) < 5000 else text[:5000]  
    out  = clf(text)[0]
    score = out["score"]
    return  score if out["label"] == "POSITIVE" else -score

# 4) new hybrid get_sentiment:
def get_sentiment(text):
    # split into sentences so long reviews don’t get “washed out”
    sents = re.split(r'(?<=[.!?])\s+', text.strip())
    # get a score for each sentence
    scores = [bert_sentiment(s) for s in sents if s]
    # fall back to VADER if something goes wrong
    if not scores:
        from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
        return SentimentIntensityAnalyzer().polarity_scores(text)["compound"]
    # average them
    return sum(scores) / len(scores)


if __name__ == "__main__":
    df["sentiment"] = df["text"].apply(get_sentiment)

    # Group by week
    weekly_summary = df.groupby("week").agg(
        avg_sentiment=("sentiment", "mean"),
        avg_rating=("rating", "mean"),
        reviews_count=("text", "count")
    ).reset_index()

    # print / review summary
    print("Weekly Reviews Summary")
    for _, row in weekly_summary.iterrows():
        week = str(row["week"])
        sentiment = row["avg_sentiment"]
        rating = row["avg_rating"]
        count = row["reviews_count"]

        if sentiment > 0.3:
            mood = "Positive"
        elif sentiment < -0.3:
            mood = "Negative"
        else:
            mood = "Mixed"

        print(f"Week: {week}")
        print(f" - Avg Sentiment Score: {sentiment:.2f} (Mixed)")
        print(f" - Avg Rating: {rating:.1f}")
        print(f" - Reviews Count: {count}\n")
