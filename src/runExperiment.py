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
from sklearn.svm import SVC
# from xgboost import XGBClassifier

# -------------------------------
# Customized functions
# -------------------------------

from src.preprocessing import preprocess_text
from src.featureExtraction import get_TFIDF_features, get_BOW_features, get_WESG_features, get_WECBOW_features

# -------------------------------
# Run Experiment
# -------------------------------

def runExperiment(dataset, featureExtract, algorithm, seed, class_mode="textblob"):

    print(" - Reading dataset")
    datafile = "data/" + dataset + ".csv"
    print(datafile)
    data = pd.read_csv(datafile)
    df = data.loc[data['language'] == 'English']    
    print(df.head())

    # -------------------------------
    # Calculating Text Polabority (TextBlob)
    # -------------------------------

    reviews = df[['id', 'review']]
    dfReviews = reviews.copy()

    if class_mode == "textblob":
        print(" - Calculating Text Polarity")
        dfReviews['polarity'] = dfReviews['review'].apply(lambda tweet: TextBlob(tweet).polarity)

    # ---------------
    # Preprocessing
    # ---------------

    print(" - Preprocessing Texts")
    df2 = dfReviews.copy()
    df2['reviewText'] = df2['review'].apply(preprocess_text)

    # -------------------------------
    # Feature Extraction
    # -------------------------------

    X = None
    match featureExtract:
        case "TFIDF":
            X = get_TFIDF_features(data = df2, dataset_name = dataset)
        case "BOW":
            X = get_BOW_features(data = df2, dataset_name = dataset)
        case "WESG":
            X = get_WESG_features(data = df2, dataset_name = dataset)
        case "WECBOW":
            X = get_WECBOW_features(data = df2, dataset_name = dataset)
    X = X.toarray()
    print(X.shape)

    # -------------------------------
    # Creating labels (textbloob)
    # -------------------------------
    
    match class_mode:
        case "textblob": 
            print(" - Creating Labels (textblob)")
            y = df2['polarity']
            ybinary = (y > 0 ) * 1
        case "scores":
            print(" - Creating Labels (scores)")
            y = df['score'].value_counts().sort_index(ascending=False)
            ybinary = (y.index >= 5).astype(int)

    # -------------------------------
    # Learning process
    # -------------------------------
  
    classifier = None
    match algorithm:
        case "KNN":
            classifier = KNeighborsClassifier()
        case "DT":
            classifier = DecisionTreeClassifier(random_state=seed)
        case "RF":
            classifier = RandomForestClassifier(random_state=seed)
        case "MNB":
            classifier = MultinomialNB()
        case "GNB":
            classifier = GaussianNB()
        case "SVM":
            classifier = SVC()
        # case "GXB":
            # classifier = XGBClassifier()
 
    print(" - Training: " + algorithm)
    
    # Stratified 10-fold CV
    skf = StratifiedKFold(n_splits=10, random_state=seed, shuffle=True)

    scores = cross_val_score(classifier, X, ybinary, cv=skf, scoring='balanced_accuracy')
    print("Results: ")
    print(scores)

    print("Mean = "+ str(scores.mean()))
    print("Sd = "+ str(scores.std()))
  
    ypred  = cross_val_predict(classifier, X, ybinary, cv=skf)

    return (scores, ypred)

# -------------------------------
# -------------------------------