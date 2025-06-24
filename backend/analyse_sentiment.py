# analyse_sentiment.py

import re
from load_data import load_data
from analyse_reviews import get_sentiment
from analyse_mixed_reviews import split_mixed_reviews
import spacy
from spacy.lang.en.stop_words import STOP_WORDS
from string import punctuation
from heapq import nlargest

df = load_data("reviews")
if 'sentiment' not in df.columns:
    df['sentiment'] = df['text'].apply(get_sentiment)

def print_weekly_metrics(df, week):
    wdf = df[df["week"] == week]
    avg_sent   = wdf["sentiment"].mean()
    avg_rating = wdf["rating"].mean()
    count      = len(wdf)
    print(f"Week {week}: avg_sentiment={avg_sent:.2f}, avg_rating={avg_rating:.1f}, reviews={count}")

def summarise_weekly_sentiment(text_list, percentage=0.3):
    try:
        nlp = spacy.load('en_core_web_sm')
    except OSError:
        return "Error: spaCy English model not found. Run: python -m spacy download en_core_web_sm"

    all_text = " ".join(text_list)
    doc = nlp(all_text)

    freq_of_word = {}
    for word in doc:
        w = word.text.lower()
        if w not in STOP_WORDS and w not in punctuation:
            freq_of_word[word.text] = freq_of_word.get(word.text, 0) + 1

    if not freq_of_word:
        return "No meaningful content to summarize"

    max_freq = max(freq_of_word.values())
    for word in freq_of_word:
        freq_of_word[word] /= max_freq

    sent_scores = {}
    for sent in doc.sents:
        sent_scores[sent] = sum(freq_of_word.get(w.text, 0) for w in sent)

    num_sentences = max(1, int(len(sent_scores) * percentage))
    top_sentences = nlargest(num_sentences, sent_scores, key=sent_scores.get)
    top_sentences.sort(key=lambda x: x.start)

    return " ".join(s.text for s in top_sentences)

if __name__ == "__main__":
    week = df["week"].unique()[2]

    # 1) Print overall metrics
    print_weekly_metrics(df, week)

    # 2) Bucket reviews into positive/negative/mixed
    positive_phrases = []
    negative_phrases = []
    mixed_reviews    = []

    for _, review in df[df["week"] == week].iterrows():
        text = review["text"]
        comp = review["sentiment"]
        clauses = split_mixed_reviews(text)
        labels  = {lbl for lbl,_ in clauses}

        if labels == {'positive', 'negative'}:
            mixed_reviews.append(text)
            for lbl, c in clauses:
                (positive_phrases if lbl=='positive' else negative_phrases).append(c)
        elif comp >=  0.5:
            positive_phrases.append(text)
        elif comp <= -0.5:
            negative_phrases.append(text)
        else:
            mixed_reviews.append(text)
            for lbl, c in clauses:
                (positive_phrases if lbl=='positive' else negative_phrases).append(c)

    # 3) Generate summaries
    summary_neg = summarise_weekly_sentiment(negative_phrases)
    summary_pos = summarise_weekly_sentiment(positive_phrases)

    print("\nAreas to improve:\n", summary_neg)
    print("\nPositive areas:\n", summary_pos)
