import pandas as pd
import numpy as np

a = pd.read_csv("H:/IC_2023_2024/Datasets/user_reviews_g1_created.csv")
b = a['review']
tamanhos = []
print(b)
count = 0
for i in b:
    print(type(i))
    print(count)
    if type(i) == int:
        i = str(i)
    tamanhos.append(len(i))  
    count+=1 
print(tamanhos)
print(max(tamanhos), min(tamanhos))

c = pd.read_csv("H:/IC_2023_2024/Datasets/user_reviews_g2_created.csv")
d = c['review']
tamanhos2 = []
print(d)
count = 0
for i in d:
    print(type(i))
    print(count)
    if type(i) == int:
        i = str(i)
    tamanhos2.append(len(i))  
    count+=1 
print(tamanhos2)
print(max(tamanhos2), min(tamanhos2))


#textmm = a.loc[maior]
#max_text_length = len(a.loc[maior, 'review'])