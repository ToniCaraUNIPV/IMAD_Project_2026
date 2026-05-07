import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
from xgboost import XGBRegressor
from sklearn.model_selection import RandomizedSearchCV
from catboost import CatBoostRegressor
import matplotlib.pyplot as plt





ds = pd.read_csv('Dataset_random.csv')
consumo = ds['LOAD']
dati = ds.drop(columns = ['LOAD'])

dati['oraSin'] = np.sin(dati.iloc[:, 0] * (2 * np.pi / 24))
dati['oraCos'] = np.cos(dati.iloc[:, 0] * (2 * np.pi / 24))


xTemp, xTest, yTemp, yTest = train_test_split(dati, consumo, test_size = 0.15, random_state = 42)
xTrain, xVal, yTrain, yVal = train_test_split(xTemp, yTemp, test_size = 0.18, random_state = 42)



# Modello
model_cat = CatBoostRegressor(
    iterations = 9600, 
    learning_rate = 0.1,
    depth = 7,
    l2_leaf_reg = 12,
    random_strength = 1,
    loss_function = 'RMSE', 
    eval_metric = 'MAE',
    random_state = 42,
    verbose = 1000 
)

model_cat.fit(xTrain, yTrain, eval_set = [(xVal, yVal)], early_stopping_rounds = 50)


print(mean_absolute_error(yTest, model_cat.predict(xTest)))
print(r2_score(yTest, model_cat.predict(xTest)))
print(mean_squared_error(yTest, model_cat.predict(xTest)))


# Con dataSet randomizzato
# 6.551427297742921
# 0.9658005041735562
# 75.74287435771289


# Grafici
ore = 7 * 24


plt.figure(figsize = (12, 7))

plt.plot(range(ore), yTest[:ore], label = 'modello reale', color = 'blue')
plt.plot(range(ore), model_cat.predict(xTest)[:ore], label = 'modello Cat', color = 'red', linestyle = '--')
plt.xlabel('ore')
plt.ylabel('consumo')
plt.title('Confronto modello')
plt.legend()

plt.show()