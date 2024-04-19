import nltk
import re
import nltk.corpus
from nltk.corpus import brown
from collections import Counter
from nltk.corpus import stopwords

def collect_corpus():
    categories = ['news', 'editorial', 'reviews', 'hobbies', 'science_fiction']
    words = brown.words(categories=categories)
    corpus = ' '.join(words)
    return corpus


# Clean the text by removing special characters, punctuation and converting to lowercase
def prepare_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)

    return text

# Create an N-gram model with frequency counts
def create_ngram_model(text, n):
    words = text.split()
    ngrams = Counter(zip(*[words[i:] for i in range(n)]))
    return ngrams


# Calculate the probability of a word given its context
def calculate_probability(ngram_model, n_minus1_gram, context,
                          word):
    context_frequency = n_minus1_gram[context]
    word_frequency = ngram_model[context + (word,)]

    if (word_frequency != 0 and context_frequency != 0):
        probability = word_frequency / context_frequency
    else:
        probability = 0

    return probability

# Predict the most likely word given a context
def predict_word(ngram_model, n_minus1_gram, context, top_n=6):
    words = [word for word in ngram_model if word[:-1] == context]
    if not words:
        return None

    # Sort words based on probability in descending order
    sorted_words = sorted(words, key=lambda x: calculate_probability(ngram_model, n_minus1_gram, context[:-1], x[-1]),
                          reverse=True)

    # Take the top N predictions
    top_predictions = sorted_words[:top_n]
    return top_predictions

# Test Trigram model with predictions
def build_model():
    corpus = collect_corpus()
    preprepared_text = prepare_text(corpus)
    n = 3  # Trigram model
    context = ('considering', 'the')
    context = tuple(word.lower() for word in context)
    ngram_model = create_ngram_model(preprepared_text, n)
    n_minus1_gram = create_ngram_model(preprepared_text, n - 1)
    predictions = predict_word(ngram_model, n_minus1_gram, context)