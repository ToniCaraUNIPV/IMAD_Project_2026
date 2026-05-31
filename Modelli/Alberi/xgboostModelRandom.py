import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
from xgboost import XGBRegressor
import matplotlib.pyplot as plt




ds = pd.read_csv('Dataset_random.csv')
consumo = ds['LOAD']
dati = ds.drop(columns = ['LOAD'])

dati['oraSin'] = np.sin(dati.iloc[:, 0] * (2 * np.pi / 24))
dati['oraCos'] = np.cos(dati.iloc[:, 0] * (2 * np.pi / 24))



xTemp, xTest, yTemp, yTest = train_test_split(dati, consumo, test_size = 0.15, random_state = 42)
xTrain, xVal, yTrain, yVal = train_test_split(xTemp, yTemp, test_size = 0.18, random_state = 42)





# XGBOOST model
xg = XGBRegressor(
    n_estimators = 5000, #5000
    max_depth = 8, #8
    subsample = 0.8, 
    colsample_bytree = 0.8,
    random_state = 42, 
    n_jobs = -1, 
    early_stopping_rounds = 20, 
    learning_rate = 0.05    #0.05
)


modelXgboost = xg.fit(xTrain, yTrain, eval_set = [(xVal, yVal)], verbose = 1000)


print(mean_absolute_error(yTest, modelXgboost.predict(xTest)))
print(r2_score(yTest, modelXgboost.predict(xTest)))
print(mean_squared_error(yTest, modelXgboost.predict(xTest)))


# Con dataSet randomizzato
# 6.46442841513062
# 0.9664212924838717
# 74.3679917796311



# Grafici
ore = 7 * 24


plt.figure(figsize = (12, 7))

plt.plot(range(ore), yTest[:ore], label = 'modello reale', color = 'blue')
plt.plot(range(ore), modelXgboost.predict(xTest)[:ore], label = 'modello XGB', color = 'red', linestyle = '--')
plt.xlabel('ore')
plt.ylabel('consumo')
plt.title('Confronto modello')
plt.legend()

plt.show()
