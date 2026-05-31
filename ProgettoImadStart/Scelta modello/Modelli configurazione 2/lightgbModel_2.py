import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
from lightgbm import LGBMRegressor
import lightgbm as lgb
import matplotlib.pyplot as plt



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



# Lightgb Model Configurazione 2
lgbModel = LGBMRegressor(
    n_estimators = 2000,
    learning_rate = 0.01,
    num_leaves = 31, #31
    max_depth = 7,
    min_child_samples = 20, #20
    colsample_bytree = 0.8,
    subsample = 0.8,
    bagging_freq = 5,
    importance_type = 'gain',
    random_state = 42,
    n_jobs = -1
)

lgbModel.fit(
    xTrain, yTrain, 
    eval_set = [(xVal, yVal)], 
    callbacks = [lgb.early_stopping(50)],
    eval_metric = 'rmse'
)


predictModel = lgbModel.predict(xTest)


print(mean_absolute_error(yTest, predictModel))
print(r2_score(yTest, predictModel))
print(mean_squared_error(yTest, predictModel))

#9.690616693711515
#0.928572814413955
#156.67773205964863


# Grafico
ore = 200

plt.figure(figsize = (12, 7))

plt.plot(range(ore), yTest[:ore], label = 'modello reale', color = 'blue')
plt.plot(range(ore), predictModel[:ore], label = 'modello LGBMR', color = 'red', linestyle = '--')
plt.xlabel('ore')
plt.ylabel('consumo')
plt.title('Confronto modello')
plt.legend()

plt.show()