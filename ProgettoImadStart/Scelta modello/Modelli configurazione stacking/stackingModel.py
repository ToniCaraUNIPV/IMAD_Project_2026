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


xTrain, xTest, yTrain, yTest = train_test_split(dataSet, consumo, test_size = 0.15, random_state = 42)

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

#9.54313492315348
#0.9304915957971764
#152.4688259831346



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