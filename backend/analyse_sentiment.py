from load_data import load_data
import spacy
from spacy.lang.en.stop_words import STOP_WORDS
from string import punctuation
from heapq import nlargest
from analyse_reviews import get_sentiment
from analyse_mixed_reviews import split_mixed_reviews

# Load data and inspect weeks
df = load_data("reviews")
if 'sentiment' not in df.columns:
    df['sentiment'] = df['text'].apply(get_sentiment)

def summarise_weekly_sentiment(text_list, percentage=0.3):
    """Summarize reviews for a given week with debugging"""
    try:
        nlp = spacy.load('en_core_web_sm')
    except OSError:
        return "Error: spaCy English model not found. Run: python -m spacy download en_core_web_sm"
    
    # # Filter reviews for the week
    # weekly_reviews = df[df["week"] == week]
    # print(f"\nFound {len(weekly_reviews)} reviews for week {week}")
    
    # if len(weekly_reviews) == 0:
    #     return "No reviews found for this week"
    
    # # Combine reviews
    all_text = " ".join(text_list)
    print("\nCombined text sample:", all_text)
    
    # Process with spaCy
    doc = nlp(all_text)
    print(f"Processed {len(doc)} tokens, {len(list(doc.sents))} sentences")
    
    # Calculate word frequencies
    freq_of_word = {}
    for word in doc:
        if word.text.lower() not in STOP_WORDS and word.text.lower() not in punctuation:
            freq_of_word[word.text] = freq_of_word.get(word.text, 0) + 1
    
    if not freq_of_word:
        return "No meaningful content to summarize"
    
    max_freq = max(freq_of_word.values())
    print(f"Most frequent word: {max(freq_of_word, key=freq_of_word.get)} (appears {max_freq} times)")
    
    # Normalize frequencies
    for word in freq_of_word:
        freq_of_word[word] /= max_freq
    
    # Score sentences
    sent_scores = {}
    for sent in doc.sents:
        score = sum(freq_of_word.get(word.text, 0) for word in sent)
        sent_scores[sent] = score
    
    # Determine how many sentences to keep
    num_sentences = max(1, int(len(sent_scores) * percentage))
    print(f"Selecting {num_sentences} sentences from {len(sent_scores)} available")
    
    # Get and sort top sentences
    top_sentences = nlargest(num_sentences, sent_scores, key=sent_scores.get)
    top_sentences.sort(key=lambda x: x.start)  # Maintain original order
    
    return " ".join(sent.text for sent in top_sentences)

# Test with actual weeks from your data
def get_sentiment_phrases(week):
    """Returns tuple of (positive_phrases, negative_phrases) for a given week"""
    weekly_reviews = df[df["week"] == week]
    
    # Initialize pure text arrays
    positive_phrases = []
    negative_phrases = []
    
    # Process each review
    for _, review in weekly_reviews.iterrows():
        text = review['text']
        sentiment = review['sentiment']
        
        if sentiment > 0.3:
            positive_phrases.append(text)
        elif sentiment < -0.3:
            negative_phrases.append(text)
        else:
            # Split mixed reviews
            for sentiment, phrase in split_mixed_reviews(text):
                if sentiment == 'positive':
                    positive_phrases.append(phrase)
                else:
                    negative_phrases.append(phrase)
    
    return positive_phrases, negative_phrases


if __name__ == "__main__":
    week = df["week"].unique()[2] 
    positive_phrases, negative_phrases = get_sentiment_phrases(week)

    print(f"\nTesting with week: {week}")
    print(f"Found {len(positive_phrases)} positive phrases and {len(negative_phrases)} negative phrases")

    summary_neg = summarise_weekly_sentiment(negative_phrases)
    summary_pos = summarise_weekly_sentiment(positive_phrases)
    
    print("\nGenerated Summary:")
    print(f"\nAreas to improve:\n{summary_neg}")
    print(f"\nPositive Areas:\n{summary_pos}")