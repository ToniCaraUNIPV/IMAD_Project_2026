import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
from catboost import CatBoostRegressor
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




# CatBoost Model

catFeauters = ['zona']

catModel = CatBoostRegressor(
    iterations = 2000,           # Numero massimo di alberi
    learning_rate = 0.03,        # Passo di apprendimento (0.01-0.05 è l'ideale qui)
    depth = 6,                   # Profondità dell'albero (simmetrico)
    l2_leaf_reg = 3,             # Regolarizzazione L2 per evitare overfitting
    random_strength = 1,         # Aggiunge casualità per esplorare più feature
    bagging_temperature = 1,     # Aiuta a gestire la varianza dei dati
    loss_function = 'RMSE',      # Funzione di perdita standard per la regressione
    eval_metric = 'MAE',         # Metrica che vogliamo monitorare durante i test
    random_seed = 42,
    verbose = 200    
)
catModel.fit(
    xTrain, yTrain, 
    cat_features = catFeauters,
    eval_set = [(xVal, yVal)], 
    early_stopping_rounds = 50, 
    use_best_model = True
)


predictModel = catModel.predict(xTest)

print(mean_absolute_error(yTest, predictModel))
print(r2_score(yTest, predictModel))
print(mean_squared_error(yTest, predictModel))

#11.468210847404814
#0.8903712153745815
#240.47411643952637



#Grafico 
ore = 100

plt.figure(figsize = (12, 7))

plt.plot(range(ore), yTest[:ore], label = 'modello reale', color = 'blue')
plt.plot(range(ore), predictModel[:ore], label = 'modello LGBMR', color = 'red', linestyle = '--')
plt.xlabel('ore')
plt.ylabel('consumo')
plt.title('Confronto modello')
plt.legend()

plt.show()
