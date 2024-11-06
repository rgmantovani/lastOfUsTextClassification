import string

# -------------------------------
# download nltk corpus (first time only)
# -------------------------------

import nltk 
nltk.download('all')

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

# -------------------------------
# Preprocessing Text
# -------------------------------

def preprocess_text(text):
    # Tokenize the text
    tokens = word_tokenize(text.lower())
    # Remove stop words
    filtered_tokens = [token for token in tokens if token not in stopwords.words('english')]
    # Lemmatize the tokens
    lemmatizer = WordNetLemmatizer()
    lemmatized_tokens = [lemmatizer.lemmatize(token) for token in filtered_tokens]
    # Join the tokens back into a string
    processed_text = ' '.join(lemmatized_tokens)
    return processed_text

# -------------------------------
# Remove Punctuation
# -------------------------------

def removePunctuation(text):
    new_text = text.translate(str.maketrans('', '', string.punctuation))
    return (new_text)

# -------------------------------
# -------------------------------
