# -------------------------------
# For consulting
# -------------------------------

# https://www.datacamp.com/tutorial/text-analytics-beginners-nltk
# https://datascienceinpractice.github.io/tutorials/18-NaturalLanguageProcessing.html
# https://medium.com/swlh/text-classification-using-tf-idf-7404e75565b8
# https://scikit-learn.org/stable/auto_examples/classification/plot_classifier_comparison.html

# -------------------------------
# -------------------------------

import pandas as pd
from textblob import TextBlob

# -------------------------------
# For learning
# -------------------------------

from sklearn.model_selection import StratifiedKFold, cross_val_score, cross_val_predict
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier

# -------------------------------
# Customized functions
# -------------------------------

from preprocessing import preprocess_text
from featureExtraction import get_TFIDF_features

# -------------------------------
# Run Experiment
# -------------------------------

def runExperiment(dataset, featureExtract, algorithm, seed):

    print(" - Reading dataset")
    data = pd.read_csv("data/" + dataset + ".csv")
    df = data.loc[data['language'] == 'English']    
    print(df.head())

    # -------------------------------
    # Calculating Text Polabority (TextBlob)
    # -------------------------------

    print(" - Calculating Text Polarity")
    reviews = df[['id', 'review']]
    dfReviews = reviews.copy()
    dfReviews['polarity'] = dfReviews['review'].apply(lambda tweet: TextBlob(tweet).polarity)

    # ---------------
    # Preprocessing
    # ---------------

    print(" - Preprocessing Texts")
    df2 = dfReviews.copy()
    df2['reviewText'] = df2['review'].apply(preprocess_text)

    # -------------------------------
    # Feature Extraction (TFIDF)
    # -------------------------------

    X = None 
    if(featureExtract == "TFIDF"):
        X = get_TFIDF_features(data = df2)
        X = X.toarray()
        print(X.shape)
   
    # -------------------------------
    # Creating labels (textbloob)
    # -------------------------------
    
    print(" - Creating Labels")
    y = df2['polarity']
    ybinary = (y > 0 ) * 1
    ybinary = ybinary.ravel()

    # -------------------------------
    # Learning process
    # -------------------------------

    # Stratified 10-fold CV
    skf = StratifiedKFold(n_splits=10, random_state=seed, shuffle=True)
  
    classifier = None
    if(algorithm == "KNN"):
        classifier = KNeighborsClassifier()
    elif(algorithm == "DT"):
        classifier = DecisionTreeClassifier(random_state=seed)
    elif(algorithm == "RF"):
        classifier = RandomForestClassifier(random_state=seed)
    elif(algorithm == "MNB"):
        classifier = MultinomialNB()
    else:
        classifier = GaussianNB()
 
    print(" - Training: " + algorithm)

    scores = cross_val_score(classifier, X, ybinary, cv=skf, scoring='balanced_accuracy')
    print("Results: ")
    print(scores)

    print("Mean = "+ str(scores.mean()))
    print("Sd = "+ str(scores.std()))
  
    ypred  = cross_val_predict(classifier, X, ybinary, cv=skf)

    return (scores, ypred)

# -------------------------------
# -------------------------------