import pandas as pd
import numpy as np

a = pd.read_csv("H:/IC_2023_2024/Datasets/relevant2.csv")
print(a)

b = a['Sentiment']
print(b)

c = 0
for i in b:
    print(i)
    c = c + i
d = c/len(b)
print(d)