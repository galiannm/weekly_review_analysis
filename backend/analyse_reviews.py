import re
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from load_data import load_data

# Load your reviews DataFrame
df = load_data("reviews")

# 1) Load a small SST-2–fine-tuned DistilBERT model once
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased-finetuned-sst-2-english")
model = AutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased-finetuned-sst-2-english")
clf = pipeline("sentiment-analysis", model=model, tokenizer=tokenizer)

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

def get_weekly_metrics(df, week):
    """Calculate metrics for a specific week"""
    wdf = df[df["week"] == week]
    return {
        'week': week,
        'avg_sentiment': wdf["sentiment"].mean(),
        'avg_rating': wdf["rating"].mean(),
        'review_count': len(wdf)
    }

def print_metrics(metrics, mood=None):
    """Print formatted metrics with optional mood"""
    print(f"\n📅 Week {metrics['week']}")
    print(f" - Avg Sentiment: {metrics['avg_sentiment']:.2f}" + 
          (f" ({mood})" if mood else ""))
    print(f" - Avg Rating: {metrics['avg_rating']:.1f} ⭐")
    print(f" - Review Count: {metrics['review_count']}")

def print_weekly_metrics(df, week):
    """Print metrics for a single week"""
    metrics = get_weekly_metrics(df, week)
    print_metrics(metrics)

def print_all_weekly_metrics(df):
    """Print metrics for all weeks with mood analysis"""
    weekly_data = []
    for week in df["week"].unique():
        metrics = get_weekly_metrics(df, week)
        metrics['mood'] = get_sentiment_category(metrics['avg_sentiment'])
        weekly_data.append(metrics)
    
    print("Weekly Reviews Summary")
    for metrics in sorted(weekly_data, key=lambda x: x['week']):
        print_metrics(metrics, metrics['mood'])

# Helper function (define this elsewhere)
def get_sentiment_category(score):
    if score > 0.3: return "Positive"
    elif score < -0.3: return "Negative"
    return "Mixed"