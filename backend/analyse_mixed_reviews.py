import re
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from load_data import load_data
from analyse_reviews import get_sentiment

analyzer = SentimentIntensityAnalyzer()

# Pre-compile regexes once
sentence_re   = re.compile(r'[^.!?]+[.!?]?')                       # grabs full sentences, keeps punctuation
contrast_re   = re.compile(r'\b(?:but|however|though|although|yet)\b',
                           flags=re.IGNORECASE)                    # non-capturing!
punct_only_re = re.compile(r'^[\W_]+$')                            # purely punctuation/spaces

def split_mixed_reviews(text):
    """
    Break a “mixed” review into smaller clauses, drop noise,
    then label each clause positive or negative.
    """
    fragments = []
    # 1) get sentences (keeps trailing .!? if present)
    sentences = sentence_re.findall(text.strip())

    for sent in sentences:
        # 2) split on contrast words, dropping the word itself
        parts = contrast_re.split(sent)
        for part in parts:
            clause = part.strip()
            # 3) drop anything empty, too short, or pure punctuation
            if not clause or len(clause) < 3 or punct_only_re.match(clause):
                continue
            # 4) ensure it ends in punctuation for clarity
            if clause[-1] not in '.!?':
                clause += '.'
            # 5) score with VADER
            score = analyzer.polarity_scores(clause)['compound']
            if score >  0.05:
                fragments.append(('positive', clause))
            elif score < -0.05:
                fragments.append(('negative', clause))
            # neutrals are ignored
    return fragments


df = load_data("reviews")
if 'sentiment' not in df.columns:
    df['sentiment'] = df['text'].apply(get_sentiment)

positive_phrases = []
negative_phrases = []
mixed_reviews   = []

week = df["week"].unique()[2]
for _, review in df[df["week"] == week].iterrows():
    text = review['text']
    comp = review['sentiment']

    # 1) First, break into clause-level sentiments
    clauses = split_mixed_reviews(text)
    labels  = {lbl for lbl, _ in clauses}

    # 2) If we see both positive AND negative, force “mixed”
    if 'positive' in labels and 'negative' in labels:
        mixed_reviews.append(text)
        for lbl, clause in clauses:
            if lbl == 'positive':
                positive_phrases.append(clause)
            else:
                negative_phrases.append(clause)

    # 3) Otherwise fall back to your strong-polarity thresholds
    elif comp >= 0.3:
        positive_phrases.append(text)
    elif comp <= -0.3:
        negative_phrases.append(text)
    else:
        # truly neutral/mixed but with no clear opposing clauses
        mixed_reviews.append(text)
        for lbl, clause in clauses:
            if lbl == 'positive':
                positive_phrases.append(clause)
            else:
                negative_phrases.append(clause)

print("Mixed reviews:", mixed_reviews)
print("\nPositive snippets:", positive_phrases)
print("\nNegative snippets:", negative_phrases)
