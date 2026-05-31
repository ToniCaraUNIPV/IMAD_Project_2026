import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
from xgboost import XGBRegressor
from sklearn.model_selection import RandomizedSearchCV
from lightgbm import LGBMRegressor
import lightgbm as lgb
import matplotlib.pyplot as plt

# ranker e classifier



ds = pd.read_csv('Dataset_random.csv')
consumo = ds['LOAD']
dati = ds.drop(columns = ['LOAD'])

dati['oraSin'] = np.sin(dati.iloc[:, 0] * (2 * np.pi / 24))
dati['oraCos'] = np.cos(dati.iloc[:, 0] * (2 * np.pi / 24))



xTemp, xTest, ytemp, yTest = train_test_split(dati, consumo, test_size = 0.15, random_state = 42)
xTrain, xVal, yTrain, yVal = train_test_split(xTemp, ytemp, test_size = 0.18, random_state = 42)



# Modello
model_lgbm = LGBMRegressor(
    n_estimators = 5000, #5000
    random_state = 42, 
    learning_rate = 0.05,
    num_leaves = 50,
    n_jobs = -1
)

model_lgbm.fit(
    xTrain, yTrain,
     eval_set = [(xVal, yVal)], 
     callbacks = [lgb.early_stopping(10)],
    eval_metric = 'rmse'
    )


print(mean_absolute_error(yTest, model_lgbm.predict(xTest)))
print(r2_score(yTest, model_lgbm.predict(xTest)))
print(mean_squared_error(yTest, model_lgbm.predict(xTest)))


# Con dataSet randomizato
# 6.731502192907194
# 0.964420133651478
# 78.8000314442222



# Grafici
ore = 7 * 24


plt.figure(figsize = (12, 7))

plt.plot(range(ore), yTest[:ore], label = 'modello reale', color = 'blue')
plt.plot(range(ore), model_lgbm.predict(xTest)[:ore], label = 'modello XGB', color = 'red', linestyle = '--')
plt.xlabel('ore')
plt.ylabel('consumo')
plt.title('Confronto modello')
plt.legend()

plt.show()