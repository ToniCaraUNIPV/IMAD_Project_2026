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


dataSet['s14'] = dati.iloc[:, 15]
dataSet['s9'] = dati.iloc[:, 10]


xTemp, xTest, yTemp, yTest = train_test_split(dataSet, consumo, test_size = 0.15, random_state = 42)
xTrain, xVal, yTrain, yVal = train_test_split(xTemp, yTemp, test_size = 0.18, random_state = 42)




# CatBoost Model configurazione 2

catModel = CatBoostRegressor(
    iterations = 3800,         
    learning_rate = 0.03,       
    depth = 6,                   
    l2_leaf_reg = 3,           
    random_strength = 1,        
    bagging_temperature = 1,     
    loss_function = 'RMSE',      
    eval_metric = 'MAE',         
    random_seed = 42,
    verbose = 200,
    thread_count = -1
)
catModel.fit(
    xTrain, yTrain,
    eval_set = [(xVal, yVal)], 
    early_stopping_rounds = 50, 
    use_best_model = True
)

predictModel = catModel.predict(xTest)



print(mean_absolute_error(yTest, predictModel))
print(r2_score(yTest, predictModel))
print(mean_squared_error(yTest, predictModel))


#9.567783025800685
#0.9302946488629594
#152.90083515073715




#Grafico 
ore = 200

plt.figure(figsize = (12, 7))

plt.plot(range(ore), yTest[:ore], label = 'modello reale', color = 'blue')
plt.plot(range(ore), predictModel[:ore], label = 'modello LGBMR', color = 'red', linestyle = '--')
plt.xlabel('ore')
plt.ylabel('consumo')
plt.title('Confronto modello')
plt.legend()

plt.show()
