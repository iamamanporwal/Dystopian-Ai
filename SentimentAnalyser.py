import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer

# Download VADER lexicon if not already downloaded
nltk.download('vader_lexicon')

# Initialize the SentimentIntensityAnalyzer
sia = SentimentIntensityAnalyzer()

def analyze_sentiment(text):
    # Get sentiment scores
    sentiment_scores = sia.polarity_scores(text)
    
    # Determine sentiment label
    if sentiment_scores['compound'] >= 0.05:
        sentiment_label = 'Positive'
    elif sentiment_scores['compound'] <= -0.05:
        sentiment_label = 'Negative'
    else:
        sentiment_label = 'Neutral'
    
    return sentiment_label, sentiment_scores

# Take user input for text
user_input = input("Enter the text you want to analyze: ")

# Perform sentiment analysis
sentiment, scores = analyze_sentiment(user_input)

print("Sentiment:", sentiment)
print("Positive:", scores['pos'])
print("Negative:", scores['neg'])
print("Neutral:", scores['neu'])
print("Compound:", scores['compound'])
