import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import RidgeCV, ElasticNetCV
from sklearn.ensemble import StackingRegressor
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
from xgboost import XGBRegressor
from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor
import lightgbm as lgb
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
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



# Scalo le varie colonne
scaler = StandardScaler()
xTrain_scaled = scaler.fit_transform(xTrain)
xVal_scaled = scaler.transform(xVal)
xTest_scaled = scaler.transform(xTest)




# Modelli 
xgModel = XGBRegressor(
    n_estimators = 2000,
    learning_rate = 0.01, 
    max_depth = 6, 
    colsample_bytree = 0.8,
    subsample = 0.7,
    random_state = 42,
    n_jobs = -1,
    early_stopping_rounds = 70
)
xgModel.fit(xTrain_scaled, yTrain, eval_set = [(xVal_scaled, yVal)], verbose = 100)



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
    xTrain_scaled, yTrain, 
    eval_set = [(xVal_scaled, yVal)], 
    callbacks = [lgb.early_stopping(50)],
    eval_metric = 'rmse'
)




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
    xTrain_scaled, yTrain,
    eval_set = [(xVal_scaled, yVal)], 
    early_stopping_rounds = 50, 
    use_best_model = True
)


mpl = MLPRegressor(
    hidden_layer_sizes = (128, 64, 32),  
    activation = 'relu',
    solver = 'adam',
    alpha = 0.01,                        
    learning_rate = 'adaptive',          # Adatta la velocità di apprendimento se si blocca
    early_stopping = True,               
    n_iter_no_change = 15,             
    max_iter = 1000,                    
    random_state = 42
)
mpl.fit(xTrain_scaled, yTrain)


# Predict Model
predictXGB = xgModel.predict(xVal_scaled)
predictLGB = lgbModel.predict(xVal_scaled)
predictCat = catModel.predict(xVal_scaled)
predictRete = mpl.predict(xVal_scaled)



nuovoDataSet = pd.DataFrame({
    'xgb': predictXGB,
    'lgb': predictLGB,
    'cat': predictCat,
    'rete': predictRete
})

# MetaModel 

metaModel = RidgeCV()
metaModel.fit(nuovoDataSet, yVal)


predictMetaModel = pd.DataFrame({
    'xgb': xgModel.predict(xTest_scaled),
    'lgb': lgbModel.predict(xTest_scaled),
    'cat': catModel.predict(xTest_scaled),
    'rete': mpl.predict(xTest_scaled)
})


predictFinale = metaModel.predict(predictMetaModel)




# Prestazioni
print(mean_absolute_error(predictFinale, yTest))
print(r2_score(yTest, predictFinale))
print(mean_squared_error(predictFinale, yTest))
print("\nPesi del Meta-Modello:")
for nome, peso in zip(nuovoDataSet.columns, metaModel.coef_):
    print(f"{nome}: {peso:.4f}")


#9.564752680497081
#0.9303900121182692
#152.69165291233344


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