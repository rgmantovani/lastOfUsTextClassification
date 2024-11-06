
from sklearn.feature_extraction.text import TfidfVectorizer

# -------------------------------
# Return TFIDF Features
# -------------------------------

def get_TFIDF_features(data):
    print(" - Extracting TFIDF Features")
    corpus = data['reviewText']
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(corpus)
    return(X)

# -------------------------------
# Return BOW Features
# -------------------------------

# TODO: Implement BOW Feature Extraction Method

# -------------------------------
# Return Word Embedding Features
# -------------------------------

# TODO: Implement Word Embedding Feature Extraction Method
# https://www.geeksforgeeks.org/python-word-embedding-using-word2vec/

# -------------------------------
# -------------------------------
