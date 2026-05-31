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

dataSet['s14'] = dati.iloc[:, 15]
dataSet['s9'] = dati.iloc[:, 10]



xTemp, xTest, yTemp, yTest = train_test_split(dataSet, consumo, test_size = 0.15, random_state = 42)
xTrain, xVal, yTrain, yVal = train_test_split(xTemp, yTemp, test_size = 0.18, random_state = 42)




# XGBoost Model 2 configurazione

xgbModel = XGBRegressor(
    n_estimators = 2000,
    learning_rate = 0.01, 
    max_depth = 6, 
    colsample_bytree = 0.8,
    subsample = 0.7,
    random_state = 42,
    n_jobs = -1,
    early_stopping_rounds = 70
)
xgbModel.fit(xTrain, yTrain, eval_set = [(xVal, yVal)], verbose = 100)



predictModel = xgbModel.predict(xTest)

print(mean_absolute_error(yTest, predictModel))
print(r2_score(yTest, predictModel))
print(mean_squared_error(yTest, predictModel))


#9.652039601365237
#0.9292444709616666
#155.204432730338



ore = 200

plt.figure(figsize = (12, 7))

plt.plot(range(ore), yTest[:ore], label = 'modello reale', color = 'blue')
plt.plot(range(ore), predictModel[:ore], label = 'modello XGB', color = 'red', linestyle = '--')
plt.xlabel('ore')
plt.ylabel('consumo')
plt.title('Confronto modello')
plt.legend()

plt.show()