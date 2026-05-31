import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.feature_selection import mutual_info_regression



ds = pd.read_csv('L1_train.csv')
consumo = ds.iloc[35066:, 1]
dati = ds.iloc[35066:]
sensori = dati.iloc[:, 1:]
dati.columns = ['ora', 'consumo'] + [f'Sensore{i}' for i in range(1, 26)]
ore = ds.iloc[35066:, 0]

primeTemperature = ds.iloc[:35066, 2:27].mean(axis = 1)
oreIniziali = ds.iloc[:35066, 0]



# Mediana
medianaT = sensori.median(axis = 1)
medianaPrimeOre = ds.iloc[:35066, 2:27].median(axis = 1)

dati['medianaT'] = medianaT

# Sforzo
sforzo = abs(dati['medianaT'] - 60)
dati['sforzo'] = sforzo


# Sin e Cos
dati['oraSin'] = np.sin(dati.iloc[:, 0] * (2 * np.pi / 24))
dati['oraCos'] = np.cos(dati.iloc[:, 0] * (2 * np.pi / 24))






# Analisi grafici
plt.figure(figsize = (15, 7))

plt.subplot(2, 1, 1)
plt.plot(consumo.values, color = 'blue', linewidth = 0.5)
plt.title('Andamento del consumo')
plt.ylabel('Consumo')


plt.subplot(2, 1, 2)
sns.histplot(consumo, kde = True, color = 'green')
plt.title('Istogramma del consumo')



plt.tight_layout()
plt.show()



# Correlazioni
plt.figure(figsize = (6, 10))

sns.heatmap(sensori.corr(), cmap = 'coolwarm', vmax = 1, vmin = -1, annot = True)
plt.title('Correlazione Sensori con consumo')
plt.show()




# Grafico sensori consumo
plt.figure(figsize = (15, 7))

mediaT = sensori.mean(axis = 1)
dati['mediaT'] = mediaT

plt.subplot(2, 1, 1)
plt.scatter(mediaT.values, consumo.values, color = 'blue', linewidth = 0.5)
plt.title('Consumo in relazione alla media dei sensori')
plt.xlabel('media temperature')
plt.ylabel('Consumo')

plt.subplot(2, 1, 2)
plt.scatter(medianaT.values, consumo.values, color = 'skyblue', linewidth = 0.5)
plt.title('Consumo in relazione alla mediana dei sensori')
plt.ylabel('Consumo')
plt.xlabel('Medianda temperature')

plt.show()



# Analisi grafici mediaT iniziale e finale
plt.figure(figsize = (15, 7))

plt.subplot(4, 1, 1)
plt.plot(mediaT.values, color = 'blue', linewidth = 0.5)
plt.xlabel('orario')
plt.ylabel('media temperature')
plt.title('Andamento temperature di train')

plt.subplot(4, 1, 2)
plt.plot(primeTemperature.values, color = 'red', linewidth = 0.5)
plt.xlabel('orario')
plt.ylabel('media temperature')
plt.title('Andamento temperature di test')

plt.subplot(4, 1, 3)
plt.plot(medianaT.values, color = 'blue', linewidth = 0.5)
plt.xlabel('Orario')
plt.ylabel('Mediana')
plt.title('Andamento Mediana di Train')

plt.subplot(4, 1, 4)
plt.plot(medianaPrimeOre.values, color = 'blue', linewidth = 0.5)
plt.xlabel('Orario')
plt.ylabel('Mediana')
plt.title('Andamento Mediana di Test')

plt.show()


# Funzione per aiutare il modello
def zoneTermiche(t):
    if t < 45: return 0
    if t < 55: return 1
    if t < 65: return 2
    if t < 75: return 3
    return 4

dati['zona_temperatura'] = dati['medianaT'].apply(zoneTermiche)


# Mutual Information
mi_elementi = dati[['ora', 'medianaT', 'sforzo', 'oraSin', 'oraCos', 'zona_temperatura', 'Sensore9', 'Sensore14']]
mi_punteggio = mutual_info_regression(mi_elementi, consumo, random_state = 42)


mi_risultato = pd.Series(mi_punteggio, index = mi_elementi.columns)
mi_risultato = mi_risultato.sort_values(ascending = False)

#Grafico Mutual information
plt.figure(figsize = (10, 8))
mi_risultato.plot(kind = 'barh', color = 'skyblue')
plt.title('Importanza colonne rispetto al load con Mutual Information')
plt.xlabel('Punteggio MI')
plt.gca().invert_yaxis()
plt.show()


print('Classifica sensori per importanza')
print(mi_risultato.head(10))
