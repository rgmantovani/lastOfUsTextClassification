import pandas as pd
import numpy as np

a = pd.read_csv('H:/IC_2023_2024/accresults.csv')
print(a)

b1 = a['NBbalanced_predict']
b2 = a['RFbalanced_predict']
b3 = a['DTbalanced_predict']
results = []
desvpads = []
auxlist = []
aux = 0
for i in range(len(b1)):
    aux = aux + b1[i]
    auxlist.append(b1[i])
aux2 = aux/len(b1)
auxlist = np.array(auxlist)
desvp = np.std(auxlist)
results.append(aux2)
desvpads.append(desvp)

auxlist = []
aux = 0
for i in range(len(b2)):
    aux = aux + b2[i]
    auxlist.append(b2[i])
aux2 = aux/len(b2)
auxlist = np.array(auxlist)
desvp = np.std(auxlist)
results.append(aux2)
desvpads.append(desvp)

auxlist = []
aux = 0
for i in range(len(b3)):
    aux = aux + b3[i]
    auxlist.append(b3[i])
aux2 = aux/len(b3)
auxlist = np.array(auxlist)
desvp = np.std(auxlist)
results.append(aux2)
desvpads.append(desvp)

print(results)
print(desvpads)