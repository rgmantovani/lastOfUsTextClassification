import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.cluster import KMeans
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier 
from sklearn.linear_model import LinearRegression 


df_bow = pd.read_csv("H:/IC_2023_2024/resultados_bow.csv")
df_auxx = pd.read_csv("H:/IC_2023_2024/Datasets/relevant.csv")
df_bow['Sentiment'] = df_auxx['Sentiment']
df_bow['Class'] = np.where(df_bow['Sentiment'] > 0, 1, 0)
df_bow.fillna(0, inplace=True)
df_bow = df_bow.drop('Sentiment', axis=1)
df_bow.to_csv("H:/IC_2023_2024/dataset_usr.csv")
print(df_bow)

#treino e teste
X, y = df_bow.iloc[:,:9566], df_bow.iloc[:, 9567]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)
#https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.StratifiedKFold.html
#acuracia balanceada por classe / fscore/ auc

#EXPORTAR AS PREDIÇÕES (cross_val_predict <----- e cross_val_score) gravar em csv

#métodos de machine learning
# naive bayes
print("\nNaive-Bayes\n")
gauss = GaussianNB()
y_pred = gauss.fit(X_train, y_train).predict(X_test)
print("Predição:\n", y_pred)
acc = round(gauss.score(X_test, y_test) * 100, 3)
print("Acurácia: ", acc)

# random forest
print("\nRandom Forest\n")
forest = RandomForestClassifier(n_estimators = 100)
rf = forest.fit(X_train, y_train)
y_predrf = rf.predict(X_test)
print("Predição:\n", y_predrf)
acc2 = round(rf.score(X_test, y_test) * 100, 3)
print("Acurácia: ", acc2)

# decision tree
print("\nDecision Tree\n")
tree = DecisionTreeClassifier(max_depth=10000)
treet = tree.fit(X_train, y_train)
y_pred_tree = treet.predict(X_test)
print("Predição:\n", y_pred_tree)
acc3 = round(treet.score(X_test, y_test) * 100, 3)
print("Acurácia: ", acc3)



