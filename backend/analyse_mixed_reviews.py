# analyse_mixed_reviews.py

import re
from analyse_reviews import bert_sentiment, get_sentiment
from load_data import load_data

# Pre-compile regexes once
sentence_re   = re.compile(r'[^.!?]+[.!?]?')                       # grabs full sentences, keeps punctuation
contrast_re   = re.compile(r'\b(?:but|however|though|although|yet)\b', flags=re.IGNORECASE)
punct_only_re = re.compile(r'^[\W_]+$')                            # purely punctuation/spaces

def split_mixed_reviews(text: str):
    """
    Break a “mixed” review into smaller clauses, drop noise,
    then label each clause positive or negative via BERT.
    Returns a list of (label, clause) tuples.
    """
    fragments = []
    sentences = sentence_re.findall(text.strip())

    for sent in sentences:
        parts = contrast_re.split(sent)
        for part in parts:
            clause = part.strip()
            if not clause or len(clause) < 3 or punct_only_re.match(clause):
                continue
            if clause[-1] not in '.!?':
                clause += '.'
            score = bert_sentiment(clause)
            if score >  0:
                fragments.append(('positive', clause))
            elif score <  0:
                fragments.append(('negative', clause))
    return fragments

# Load data and annotate if needed
df = load_data("reviews")
if 'sentiment' not in df.columns:
    df['sentiment'] = df['text'].apply(get_sentiment)

positive_phrases = []
negative_phrases = []
mixed_reviews   = []

week = df["week"].unique()[1]
for _, review in df[df["week"] == week].iterrows():
    text = review['text']
    comp = review['sentiment']

    clauses = split_mixed_reviews(text)
    labels  = {lbl for lbl, _ in clauses}

    # 1) Force mixed if both polarities exist
    if labels == {'positive', 'negative'}:
        mixed_reviews.append(text)
        for lbl, clause in clauses:
            if lbl == 'positive':
                positive_phrases.append(clause)
            else:
                negative_phrases.append(clause)

    # 2) Purely positive or negative by strong overall sentiment
    elif comp >= 0.5:
        positive_phrases.append(text)
    elif comp <= -0.5:
        negative_phrases.append(text)

    # 3) Everything else is mixed
    else:
        mixed_reviews.append(text)
        for lbl, clause in clauses:
            if lbl == 'positive':
                positive_phrases.append(clause)
            else:
                negative_phrases.append(clause)
