import pandas as pd #para uso dos datasets e dataframes
import textblob #para polarizador
import numpy as np #atribuição numérica
import matplotlib.pyplot as plt #gráficos
import seaborn as sns
import collections #elaboração da WordCloud
from PIL import Image
from wordcloud import WordCloud, ImageColorGenerator, STOPWORDS
import nltk #tratamento dos dados
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem.snowball import SnowballStemmer
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.classify import NaiveBayesClassifier
from nltk.corpus import subjectivity
from nltk.sentiment import SentimentAnalyzer
from nltk.sentiment.util import *
import re
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.decomposition import NMF, PCA
from sklearn.manifold import TSNE
import spacy
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')

def exct(a, target):#excludeterm
  target = target.replace(a, '')
  return target

def cloudWordgen(stopw, dataframe): #elaboração das cloudwords
  wc = WordCloud(stopwords = stopw, background_color = "black", width=5000, height=1100).generate(dataframe)
  fig, ax = plt.subplots(figsize=(10,6))
  ax.imshow(wc,interpolation='bilinear')
  ax.set_axis_off()
  plt.imshow(wc)
  wc.to_file("img.png")

def sw(df, dftype): #stopwords da bilioteca
  stops = set(stopwords.words(dftype))
  stops2 = []
  for s in stops:
    s = " " + s + " "
    stops2.append(s)
  for s in df:
    if s in stops2:
      df = df.remove(s)
  return df

def tokenization (df): #elaboração dos tokens
  s = str(word_tokenize(df))
  return s

def stemming(df, dftype): #stemização
  snowball = SnowballStemmer(language = dftype)
  df = str(snowball.stem(df))
  return df

def preprocessor (df, dftype):
  data = sw(df, dftype)
  data = tokenization(data)
  data = stemming(df, dftype)
  return data

def geraCW(df, df_2, labels, labelr, labelsn, stops2, dfwords2):
  words_scoring_pos = []
  wsp = []
  words_scoring_neg = []
  wsn = []

  for i in range(0, len(df)-1):
    sc = df.at[i, labels]
    if i == 1:
      print(sc, labels)
    data = df.at[i, labelr]
    list_generative = [data, sc]
    lis_pal = data
    if sc > 50:
      words_scoring_pos.append(list_generative)
      wsp.append(lis_pal)

    elif sc <= 50:
      words_scoring_neg.append(list_generative)
      wsn.append(lis_pal)

  print(words_scoring_pos)
  print(words_scoring_neg)
  print(wsp)
  print(wsn)

#preprocpos
#positivo
  dfwscp = " ".join(s for s in wsp)
  for s in un_signals:
      dfwscp = exct(s, dfwscp)
  dfwscp = preprocessor(dfwscp, 'english')
  dfwscp2 = ""
  for t in un_signals:
      dfwscp2 = dfwscp.replace(t, " ")
  dfwscp2= re.sub(r"[-()\[\]\'\"],", "", dfwscp2)


  print("\n",dfwscp2)

  if dfwscp2 != "":
    cloudWordgen(stops2, dfwscp2)
  else:
    print("Lista vazia - Scoring")

#negativo
  dfwscn = " ".join(s for s in wsn)
  for s in un_signals:
    dfwscn = exct(s, dfwscn)
  dfwscn = preprocessor(dfwscn, 'english')
  dfwscn2 = ""
  for t in un_signals:
      dfwscn2 = dfwscn.replace(t, " ")
  dfwscn2 = re.sub(r"[-()\[\]\'\"],", "", dfwscn2)



  print("\n",dfwscn2)

  if dfwscn2 != "":
    cloudWordgen(stops2, dfwscn2)
  else:
    print("Lista vazia - Scoring")


#por sentimento

  words_sent_pos = []
  words_sent_neg = []
  wssp = []
  wssn = []

  for i in range(0, len(df_2)-1):
    sc = df.at[i, labelsn]
    data = df.at[i, labelr]
    list_generative = [data, sc]
    lis_pal = data
    if sc > 0:
      words_sent_pos.append(list_generative)
      wssp.append(lis_pal)
    elif sc <= 0:
      words_sent_neg.append(list_generative)
      wssn.append(lis_pal)

  print("Positivos (Scoring):\n", words_sent_pos)
  print("Negativos (Scoring):\n", words_sent_neg)
  print("Positivos (Sentiment):\n", wssp)
  print("Negativos (Sentiment):\n", wssn)

  #positivo
  dfwsscp = " ".join(s for s in wssp)
  for s in un_signals:
      dfwsscp = exct(s, dfwsscp)
  dfwsscp = preprocessor(dfwsscp, 'english')
  dfwsscp2 = ""
  for t in un_signals:
      dfwsscp2 = dfwsscp.replace(t, " ")
  dfwsscp2= re.sub(r"[-()\[\]\'\"],", "", dfwsscp2)


  print("\n",dfwsscp2)

  if dfwsscp2 != "":
    cloudWordgen(stops2, dfwsscp2)
  else:
    print("Lista vazia (pos) - Sentiment")

#negativo
  dfwsscn = " ".join(s for s in wsn)
  for s in un_signals:
    dfwsscn = exct(s, dfwsscn)
  dfwsscn = preprocessor(dfwsscn, 'english')
  dfwsscn2 = ""
  for t in un_signals:
      dfwsscn2 = dfwsscn.replace(t, " ")
  dfwsscn2 = re.sub(r"[-()\[\]\'\"],", "", dfwsscn2)



  print("\n",dfwsscn2)

  if dfwsscn2 != "":
    cloudWordgen(stops2, dfwsscn2)
  else:
    print("Lista vazia (neg) - Sentiment")

#------------------------------------------------------------------------
#------------------------------------------------------------------------
#------------------------------------------------------------------------
#a gambiarra suprema
def LimpaDataset(df, filter, csv_address):
  df = df[df['language'] == filter]

  for i in range(len(df['review'])):
    df.iloc[i, df.columns.get_loc("review")] = df.iloc[i, df.columns.get_loc("review")].lower()
    df.iloc[i, df.columns.get_loc("review")] = re.sub(r"[-()\[\]\'\"],.", "", df.iloc[i, df.columns.get_loc("review")])
  df.to_csv(csv_address)
  print("EEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEE\n")
  print(df)
  return df
#------------------------------------------------------------------------
#------------------------------------------------------------------------
def DatasetTratado(df, stopwords, columns_remove, columns_objective, csv_address):
  #remover colunas indesejadas
  for i in columns_remove:
    if i in df:
      df = df.drop(i, axis = 1)

  #inserir coluna de sentimento
  df_alt = df
  df_s = []
  index_column = df.columns.get_loc(columns_objective)
  df['Sentiment'] = np.where(df['score'] > 50, 1, 0)

  #achar a(s) coluna(s) objetiva(s) do dataframe para criar a string
  for i in df.columns:
    for k in range(0, len(df)-1):
      df_s.append([df.iloc[k, index_column], df.iloc[k, df.shape[1] - 1]])
    relevant = index_column

  #agora para a versão com polarização
  df_alt['Sentiment'] = np.where(df['score'] > 0, 1, 1)
  df_s2 = []
  df_s3 = []
  for k in range(0, len(df)):
      w = df.iloc[k, index_column]
      aux = textblob.TextBlob(w)
      aux2 = aux.sentiment
      df_s2.append([df.iloc[k, index_column], aux2.polarity, aux2.subjectivity])
      df_s3.append(aux2.polarity)
  relevant2 = index_column
  print("Sentiment\n")
  print(df_s3)
  print(len(df_s3))

  df_alt['Sentiment'] = df_s3
  #df_alt['review'] = df_alt['review'].apply(lambda x: preprocessor_lambda(x, 'english')) - - - - - - - - - - - -
  df_alt.to_csv(csv_address)

  #tratamento das palavras
  a = list(df.columns)
  if relevant == relevant2:
    print("A classe relevante é: ", a[relevant], "\n\n")
  else:
    print("Algo deu errado!!!\n")

  words = df.iloc[:, relevant]
  dfwords = " ".join(s for s in words)
  palavras_dfwords = dfwords.split(' ')
  print("As palavras são:\n", palavras_dfwords, "\n")
  print("Tamanho:\n" + str(len(palavras_dfwords)))

  #lqp = []
  #for s in palavras_dfwords:
  #  a = palavras_dfwords.count(s)
  #  lqp.append([s, a])

  dfwords = preprocessor(dfwords, 'english')
  print(type(dfwords))
  print(dfwords)

  dfwords2 = ""
  for t in un_signals:
      dfwords2 = dfwords.replace(t, " ")

  dfwords2 = re.sub(r"[-()\[\]\'\"],", "", dfwords2)
  dfwords2 = dfwords2.lower()

  geraCW(df, df_alt, 'score', 'review', 'Sentiment', stopwords, dfwords2)
  #cloudWordgen(stopwords, dfwords2)
  return [df_alt, dfwords2]


#---------------------------------------------------------------------------------------#
#---------------------------------------------------------------------------------------#
#---------------------------------------------------------------------------------------#
#stopwords
sww = set(STOPWORDS)
lista_stopwords = [" and ", " is ", " was ", " were ", " are ", " am ", " I ", " you ", " he ", " she ", " it ", " a ", " or ", " when ", " where ", " how ", " we ", " they ", " me ", " mine ", " myself ", " last ", " action ", " game ", " world ", " things ", " already ", " played ", " us "]
lista_stopwords.extend(["edition", "remastered", "naughty", "dog", "playstation", "1080p", "PS4", "Sony", "'ve ", "textures", "whether", "PS3", "last", "us", "game", "played", "remaster", "one", "version", "play", "and", "dlc", "look", "consider", "still"])
sww.update(lista_stopwords)
stops2 = list(stopwords.words('english'))
un_signals = ['-', '.', ',', ';', '!', '?', '(', ')', '[', ']', '{', '}', '\'', '\"']
stops2 = stops2 + lista_stopwords + un_signals

def bagOfWords(text):
  text2 = word_tokenize(text)
  bag_of_words = {word: text2.count(word) for word in set(text2)}
  bow_vector = []
  for i in text2:
    bow_vector.append(text2.count(i))
  #print('Bag of words:')
  #print(list(bag_of_words.items())[:])
  return [bag_of_words, bow_vector]


#tf-idf
def tfIDF(text):
  # TF-IDF
  tfidf_vec = TfidfVectorizer()
  X_tfidf = tfidf_vec.fit_transform([text])
  print(type(X_tfidf))
  print(X_tfidf)
  #print('TF-IDF:')
  #print(tfidf_vec.get_feature_names_out()[:])
  #print(X_tfidf.toarray()[:][:])
  #return X_tfidf.toarray()
  return [tfidf_vec.get_feature_names_out()[:], X_tfidf.toarray()[:][:]]

#word embedding
#def wordEmbedding():

#hashing vectorizer

#pca



def featureExtraction(dataset, text_all, column): #função principal que irá retornar as funcionalidades
  texto = dataset[column]
  texto2 = text_all
  count = 0
  for i in texto:
    i = preprocessor(i, 'english')
  lista_metricas = []
  aux = []
  lm2 = []
  #bow
  print("\nSeção: Bag of Words\n")
  for i in texto:
    aj = bagOfWords(i)
    aux.append(aj[0]) #por linha
    lm2.append(aj[1])
  print('\nPara todos!!!\n')
  #bagOfWords(texto2) #todo o texto do dataframe
  #tfidf
  lista_metricas.append(aux)
  aux2 = []
  print("\nSeção: TF-IDF\n")
  for i in texto:
    #print(i, " ", tfIDF(i))
    aux2.append(tfIDF(i)) #por linha
    if count < 5:
      print(aux2[count][1])
      count+=1
  print('\nPara todos!!!\n')
  #tfIDF(texto2) #todo o texto do dataframe
  lista_metricas.append(aux2)
  return [lista_metricas, lm2]

a1 = pd.read_csv('H:/IC_2023_2024/Datasets/relevant.csv')
a2 = pd.read_csv()
b1 = pd.read_csv('H:/IC_2023_2024/Datasets/relevant2.csv')
b2 = pd.read_csv()


feattest = featureExtraction(a1, a2, 'review')
ft1 = feattest[0]
ft2 = feattest[1]

feattest2 = featureExtraction(b1, b2, 'review')
ft1_2 = feattest2[0]
ft2_2 = feattest2[1]

#print("Aqui --- \n\n\n")
#print(feattest[1][0])
dataframe_resultsbow = pd.DataFrame(ft1[:][0])
dataframe_resultstfidf = pd.DataFrame(ft1[:][1])

dataframe_resultsbow2 = pd.DataFrame(ft1_2[:][0])
dataframe_resultstfidf2 = pd.DataFrame(ft1_2[:][1])
#dataframe_results['bow'] = feattest[:][0]
#dataframe_results['tfidf'] = feattest[:][1]
dataframe_rb2 = pd.DataFrame(ft2)
dataframe_rb2.to_csv('H:/IC_2023_2024/resultados_bow22.csv')

dataframe_rb22 = pd.DataFrame(ft2_2)
dataframe_rb22.to_csv('H:/IC_2023_2024/resultados_bow222.csv')

dataframe_resultsbow.to_csv('H:/IC_2023_2024/resultados_bow.csv')
dataframe_resultsbow2.to_csv('H:/IC_2023_2024/resultados_bow2.csv')
dataframe_resultstfidf2.to_csv('H:/IC_2023_2024/resultados_tfidif2.csv')
dataframe_resultstfidf.to_csv('H:/IC_2023_2024/resultados_tfidf.csv')
#dataframe_results2 = pd.DataFrame()
#dataframe_results2['bow'] = feattest[:][0]
#dataframe_results2['tfidf'] = feattest[:][1]
#dataframe_results2.to_csv('H:/IC_2023_2024/resultados_num2.csv')