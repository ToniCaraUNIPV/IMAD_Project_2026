import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import RidgeCV, ElasticNetCV
from sklearn.ensemble import StackingRegressor
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
from xgboost import XGBRegressor
from sklearn.model_selection import RandomizedSearchCV
from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor
import lightgbm as lgb
import matplotlib.pyplot as plt






ds = pd.read_csv('Dataset_random.csv')
consumo = ds['LOAD']
dati = ds.drop(columns = ['LOAD'])

dati['oraSin'] = np.sin(dati.iloc[:, 0] * (2 * np.pi / 24))
dati['oraCos'] = np.cos(dati.iloc[:, 0] * (2 * np.pi / 24))


xTemp, xTest, yTemp, yTest = train_test_split(dati, consumo, test_size = 0.15, random_state = 42)
xTrain, xVal, yTrain, yVal = train_test_split(xTemp, yTemp, test_size = 0.18, random_state = 42)





# Modelli 
xgModel = XGBRegressor(
    n_estimators = 5000, #5000
    max_depth = 8, #8
    subsample = 0.8, 
    colsample_bytree = 0.8,
    random_state = 42, 
    n_jobs = -1, 
    early_stopping_rounds = 20, 
    learning_rate = 0.05    #0.05
)
modelXgboost = xgModel.fit(xTrain, yTrain, eval_set = [(xVal, yVal)], verbose = 1000)



lgbModel = LGBMRegressor(
    n_estimators = 5000, #5000
    random_state = 42, 
    learning_rate = 0.05,
    num_leaves = 50,
    n_jobs = -1
)

lgbModel.fit(
    xTrain, yTrain,
    eval_set = [(xVal, yVal)], 
    callbacks = [lgb.early_stopping(10)],
    eval_metric = 'rmse'
)




catModel = CatBoostRegressor(
    iterations = 15800, #così è al massimo, posso tentare anche un 13000 oppure un 14000
    learning_rate = 0.1,
    depth = 7,
    l2_leaf_reg = 12,
    random_strength = 1,
    loss_function = 'RMSE', 
    eval_metric = 'MAE',
    random_state = 42,
    verbose = 1000 
)

catModel.fit(xTrain, yTrain, eval_set = [(xVal, yVal)], early_stopping_rounds = 50)




# Predict Model
predictXGB = xgModel.predict(xVal)
predictLGB = lgbModel.predict(xVal)
predictCat = catModel.predict(xVal)



nuovoDataSet = pd.DataFrame({
    'xgb': predictXGB,
    'lgb': predictLGB,
    'cat': predictCat
})

# MetaModel 

metaModel = RidgeCV()
metaModel.fit(nuovoDataSet, yVal)


predictMetaModel = pd.DataFrame({
    'xgb': xgModel.predict(xTest),
    'lgb': lgbModel.predict(xTest),
    'cat': catModel.predict(xTest)
})


predictFinale = metaModel.predict(predictMetaModel)




# Prestazioni
print(mean_absolute_error(predictFinale, yTest))
print(r2_score(yTest, predictFinale))
print(mean_squared_error(predictFinale, yTest))


#Con dataSet randomizzato
# 6.3696658605065295
# 0.9674831192008091
# 72.0163253100785



#Grafico
ore = 7 * 24

plt.figure(figsize = (12, 7))

plt.plot(range(ore), yTest[:ore], label = 'modello reale', color = 'blue')
plt.plot(range(ore), predictFinale[:ore], label = 'modello Blending', color = 'red', linestyle = '--')
plt.xlabel('ore')
plt.ylabel('consumo')
plt.title('Confronto modello')
plt.legend()

plt.show()