import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error




ds = pd.read_csv('L1_train.csv')
consumo = ds.iloc[35066:, 1]
dati = ds.iloc[35066:]

dataSet = pd.DataFrame()



#Setto dati per il modello

sensoriMediana = dati.iloc[:, 1:].median(axis = 1)
dataSet['medianaT'] = sensoriMediana

sforzo = abs(dataSet['medianaT'] - 60)
dataSet['sforzo'] = sforzo

oreGiorno = dati.iloc[:, 0] % 24
dataSet['oraSin'] = np.sin(oreGiorno * (2 * np.pi / 24))
dataSet['oraCos'] = np.cos(oreGiorno * (2 * np.pi / 24))


# Funzione per aiutare il modello
def zoneTermiche(t):
    if t < 45: return 0
    if t < 55: return 1
    if t < 65: return 2
    if t < 75: return 3
    return 4

dataSet['zona'] = dataSet['medianaT'].apply(zoneTermiche)



xTemp, xTest, yTemp, yTest = train_test_split(dataSet, consumo, test_size = 0.15, random_state = 42)
xTrain, xVal, yTrain, yVal = train_test_split(xTemp, yTemp, test_size = 0.18, random_state = 42)




# Random Forest
rfModel = RandomForestRegressor(
    n_estimators = 200,
    max_depth = 15,
    min_samples_leaf = 4,
    random_state = 42,
    n_jobs = -1
)
rfModel.fit(xTrain, yTrain)


predicModel = rfModel.predict(xVal)

print(mean_absolute_error(yVal, predicModel))
print(r2_score(yVal, predicModel))
print(mean_squared_error(yVal, predicModel))

print(rfModel.feature_importances_)