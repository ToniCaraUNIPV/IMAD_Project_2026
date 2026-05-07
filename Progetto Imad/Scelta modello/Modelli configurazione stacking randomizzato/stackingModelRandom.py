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


xTrain, xTest, yTrain, yTest = train_test_split(dati, consumo, test_size = 0.15, random_state = 42)


# Modelli
modelli = [
    ('xgb', XGBRegressor(
         n_estimators = 2000,
        learning_rate = 0.01, 
        max_depth = 6, 
        colsample_bytree = 0.8,
        subsample = 0.7,
        random_state = 42,
        n_jobs = -1
       # early_stopping_rounds = 70
    )),

    ('lgbm', LGBMRegressor(
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
    )),

    ('cat', CatBoostRegressor(
        iterations = 3800,           # Numero massimo di alberi
        learning_rate = 0.03,        # Passo di apprendimento (0.01-0.05 è l'ideale qui)
        depth = 6,                   # Profondità dell'albero (simmetrico)
        l2_leaf_reg = 3,             # Regolarizzazione L2 per evitare overfitting
        bagging_temperature = 1,     # Aiuta a gestire la varianza dei dati
        loss_function = 'RMSE',      # Funzione di perdita standard per la regressione
        eval_metric = 'MAE',         # Metrica che vogliamo monitorare durante i test
        random_seed = 42,
        verbose = 200,
        thread_count = -1
    ))
]



# Meta-Modello

metaModello = StackingRegressor(
    estimators = modelli,
    final_estimator = RidgeCV(),
    cv = 5,
    n_jobs = -1,
    passthrough = False #Il capo guarda SOLO le previsioni dei modelli, non i dati grezzi
)
metaModello.fit(xTrain, yTrain)


predictModello = metaModello.predict(xTest)



# Prestazione

print(mean_absolute_error(yTest, predictModello))
print(r2_score(yTest, predictModello))
print(mean_squared_error(yTest, predictModello))

#7.290487551708087
#0.959397011928902
#89.92492285921173



# Grafico
ore = 200

plt.figure(figsize = (12, 7))

plt.plot(range(ore), yTest[:ore], label = 'modello reale', color = 'blue')
plt.plot(range(ore), predictModello[:ore], label = 'modello Stacking', color = 'red', linestyle = '--')
plt.xlabel('ore')
plt.ylabel('consumo')
plt.title('Confronto modello')
plt.legend()

plt.show()