from sklearn.model_selection import StratifiedKFold
import pandas as pd
import numpy as np
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import balanced_accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier

# Carregando e preparando os dados
df_tfidf = pd.read_csv("H:/IC_2023_2024/resultados_tfidf.csv")
df_auxx = pd.read_csv("H:/IC_2023_2024/Datasets/relevant.csv")
df_tfidf['Sentiment'] = df_auxx['Sentiment']
df_tfidf['Class'] = np.where(df_tfidf['Sentiment'] > 0, 1, 0)
df_tfidf.fillna(0, inplace=True)
df_bow = df_tfidf.drop('Sentiment', axis=1)

print(df_bow)

# Inicializando listas para armazenar resultados
lista_nb = []
lista_nb2 = []
lista_nbc = []
lista_rf = []
lista_rf2 = []
lista_rfc = []
lista_dt = []
lista_dt2 = []
lista_dtc = []

# Separando variáveis independentes e dependentes
X, y = np.array(df_bow.iloc[:,:-1]), np.array(df_bow.iloc[:,-1])  # Ajuste para usar todas as colunas, menos a última, para X
strk = StratifiedKFold(n_splits=5, random_state=1, shuffle=True)
count = 0

for train_index, test_index in strk.split(X, y):
    print("Fold ", count)
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]

    # Naive-Bayes
    print("\nNaive-Bayes\n")
    gauss = GaussianNB()
    gauss.fit(X_train, y_train)
    y_pred = gauss.predict(X_test)
    acc = round(gauss.score(X_test, y_test) * 100, 3)
    print("Acurácia: ", acc)
    lista_nb.append(acc)

    # Usando cross_val_predict para obter as predições
    y_pred_train = cross_val_predict(gauss, X_train, y_train, cv=strk, method='predict')
    accb = balanced_accuracy_score(y_train, y_pred_train)
    y_pred_test = cross_val_predict(gauss, X_test, y_test, cv=strk, method='predict')
    accuracy = balanced_accuracy_score(y_test, y_pred_test)
    lista_nb2.append(accb)
    lista_nbc.append(accuracy)

    # Random Forest
    print("\nRandom Forest\n")
    forest = RandomForestClassifier(n_estimators=100)
    forest.fit(X_train, y_train)
    y_predrf = forest.predict(X_test)
    acc2 = round(forest.score(X_test, y_test) * 100, 3)
    print("Acurácia: ", acc2)
    lista_rf.append(acc2)

    # Usando cross_val_predict para obter as predições
    y_pred_train_rf = cross_val_predict(forest, X_train, y_train, cv=strk, method='predict')
    accb2 = balanced_accuracy_score(y_train, y_pred_train_rf)
    y_pred_test_rf = cross_val_predict(forest, X_test, y_test, cv=strk, method='predict')
    accuracy2 = balanced_accuracy_score(y_test, y_pred_test_rf)
    lista_rf2.append(accb2)
    lista_rfc.append(accuracy2)

    # Decision Tree
    print("\nDecision Tree\n")
    tree = DecisionTreeClassifier(max_depth=10000)
    tree.fit(X_train, y_train)
    y_pred_tree = tree.predict(X_test)
    acc3 = round(tree.score(X_test, y_test) * 100, 3)
    print("Acurácia: ", acc3)
    lista_dt.append(acc3)

    # Usando cross_val_predict para obter as predições
    y_pred_train_tree = cross_val_predict(tree, X_train, y_train, cv=strk, method='predict')
    accb3 = balanced_accuracy_score(y_train, y_pred_train_tree)
    y_pred_test_tree = cross_val_predict(tree, X_test, y_test, cv=strk, method='predict')
    accuracy3 = balanced_accuracy_score(y_test, y_pred_test_tree)
    lista_dt2.append(accb3)
    lista_dtc.append(accuracy3)

    count += 1

# Criando DataFrame para salvar resultados
dfr = pd.DataFrame()
dfr['NBsimple'] = lista_nb
dfr['NBbalanced'] = lista_nb2
dfr['NBbalanced_predict'] = lista_nbc
dfr['RFsimple'] = lista_rf
dfr['RFbalanced'] = lista_rf2
dfr['RFbalanced_predict'] = lista_rfc
dfr['DTsimple'] = lista_dt
dfr['DTbalanced'] = lista_dt2
dfr['DTbalanced_predict'] = lista_dtc

dfr.to_csv('H:/IC_2023_2024/accresults.csv', index=False)
