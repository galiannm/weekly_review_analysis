from load_data import load_data
from analyse_reviews import get_sentiment, print_weekly_metrics, print_all_weekly_metrics
from analyse_sentiment import get_summary_sentiment

if __name__ == "__main__":
    df = load_data("reviews")

    # Compute sentiment if missing
    if "sentiment" not in df.columns:
        df["sentiment"] = df["text"].apply(get_sentiment)

    # Choose week
    week = df["week"].unique()[3]
    print_weekly_metrics(df, week)

    summary_neg, summary_pos = get_summary_sentiment(week)

    print("\n✅ Positive Highlights:\n", summary_pos)
    print("\n⚠️  Areas to Improve:\n", summary_neg)
