import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
from xgboost import XGBRegressor
from sklearn.model_selection import RandomizedSearchCV




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



# XGBOOST Model
xgModel = XGBRegressor(
    n_estimators = 2000,
    learning_rate = 0.01, 
    max_depth = 6, 
    colsample_bytree = 0.8,
    subsample = 0.7,
    random_state = 42,
    n_jobs = -1,
    early_stopping_rounds = 70
)
xgModel.fit(xTrain, yTrain, eval_set = [(xVal, yVal)], verbose = 1000)


predictModel = xgModel.predict(xTest)


print(mean_absolute_error(yTest, predictModel))
print(r2_score(yTest, predictModel))
print(mean_squared_error(yTest, predictModel))


# Grafico
ore = 100

plt.figure(figsize = (12, 7))

plt.plot(range(ore), yTest[:ore], label = 'modello reale', color = 'blue')
plt.plot(range(ore), predictModel[:ore], label = 'modello XGB', color = 'red', linestyle = '--')
plt.xlabel('ore')
plt.ylabel('consumo')
plt.title('Confronto modello')
plt.legend()

plt.show()



#11.589304826612018
#0.8889692670903788
#243.54933319112573