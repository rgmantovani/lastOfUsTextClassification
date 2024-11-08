
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from gensim.models import Word2Vec
from scipy import sparse
from os import path
import numpy as np
import sys

# -------------------------------
# Return TFIDF Features
# -------------------------------

def get_TFIDF_features(data, dataset_name:str=None):
    feature_file = "features/tfidf_" + dataset_name + ".npz"
    # checks if feature file exists
    if(path.exists(feature_file)):
        print("- Getting features from file")
        X = sparse.load_npz(feature_file)
    else:
        print(" - Extracting TFIDF Features")
        corpus = data['reviewText']
        vectorizer = TfidfVectorizer()
        X = vectorizer.fit_transform(corpus)
        if dataset_name:
            try:
                print(" - Saving TFIDF Features to file")
                sparse.save_npz(feature_file, X)
            except Exception as e:
                print(" - File not saved: " + e)
    return(X)

# -------------------------------
# Return BOW Features
# -------------------------------

def get_BOW_features(data, dataset_name: str = None):
    feature_file = "features/bow_" + dataset_name + ".npz"
    # Checks if feature file exists
    if path.exists(feature_file):
        print("- Getting features from file")
        X = sparse.load_npz(feature_file)
    else:
        print(" - Extracting BOW Features")
        corpus = data['reviewText']
        vectorizer = CountVectorizer()
        X = vectorizer.fit_transform(corpus)
        if dataset_name:
            try:
                print(" - Saving BOW Features to file")
                sparse.save_npz(feature_file, X)
            except Exception as e:
                print(" - File not saved: " + str(e))
    return X

# -------------------------------
# Return Word Embedding Skip-Gram Features
# -------------------------------

def get_WESG_features(data, dataset_name: str = None, vector_size: int = 100, window: int = 5, min_count: int = 1):
    feature_file = "features/wesg_" + dataset_name + ".npz"
    # Checks if feature file exists
    if path.exists(feature_file):
        print("- Getting features from file")
        X = sparse.load_npz(feature_file)
    else:
        print(" - Extracting Word Embedding Skip-gram Features")
        corpus = [text.split() for text in data['reviewText']]  
        model = Word2Vec(corpus, vector_size=vector_size, window=window, min_count=min_count, sg=1, workers=4)
        embeddings = []
        for doc in corpus:
            words = [model.wv[word] for word in doc if word in model.wv]
            if words:
                avg_embedding = np.mean(words, axis=0)
            else:
                avg_embedding = np.zeros(vector_size)  
            embeddings.append(avg_embedding)
        X = sparse.csr_matrix(embeddings)
        if dataset_name:
            try:
                print(" - Saving Word Embedding Skip-gram Features to file")
                sparse.save_npz(feature_file, X)
            except Exception as e:
                print(" - File not saved: " + str(e))
    return X

# -------------------------------
# Return Word Embedding Continuous BOW Features
# -------------------------------

def get_WECBOW_features(data, dataset_name: str = None, vector_size: int = 100, window: int = 5, min_count: int = 1):
    feature_file = "features/wecbow_" + dataset_name + ".npz"
    # Checks if feature file exists
    if path.exists(feature_file):
        print("- Getting features from file")
        X = sparse.load_npz(feature_file)
    else:
        print(" - Extracting Word Embedding CBOW Features")
        corpus = [text.split() for text in data['reviewText']]  
        model = Word2Vec(corpus, vector_size=vector_size, window=window, min_count=min_count, workers=4)
        embeddings = []
        for doc in corpus:
            words = [model.wv[word] for word in doc if word in model.wv]
            if words:
                avg_embedding = np.mean(words, axis=0)
            else:
                avg_embedding = np.zeros(vector_size)  
            embeddings.append(avg_embedding)
        X = sparse.csr_matrix(embeddings)
        if dataset_name:
            try:
                print(" - Saving Word Embedding CBOW Features to file")
                sparse.save_npz(feature_file, X)
            except Exception as e:
                print(" - File not saved: " + str(e))
    return X

# -------------------------------
# -------------------------------
