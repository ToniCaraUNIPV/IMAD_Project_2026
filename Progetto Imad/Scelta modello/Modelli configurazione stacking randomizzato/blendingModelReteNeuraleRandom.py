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


#low code platform



ds = pd.read_csv('Dataset_random.csv')
consumo = ds['LOAD']
dati = ds.drop(columns = ['LOAD'])

dati['oraSin'] = np.sin(dati['TIMESTAMP'] * (2 * np.pi / 24))
dati['oraCos'] = np.cos(dati['TIMESTAMP'] * (2 * np.pi / 24))
dati = dati.drop(columns = ['TIMESTAMP'])

# 1. Media termica dei sensori (il 'clima' generale dell'istante)
#dati['temp_media'] = dati.filter(like='w').mean(axis=1)

# 2. Variabilità tra i sensori (capisce se c'è un'attività localizzata)
#dati['temp_std'] = dati.filter(like='w').std(axis=1)

# 3. Differenza tra il punto più caldo e il più freddo
#dati['temp_range'] = dati.filter(like='w').max(axis=1) - dati.filter(like='w').min(axis=1)

xTemp, xTest, yTemp, yTest = train_test_split(dati, consumo, test_size = 0.15, random_state = 42)
xTrain, xVal, yTrain, yVal = train_test_split(xTemp, yTemp, test_size = 0.18, random_state = 42)




# Scalo le varie colonne
scaler = StandardScaler()
xTrain_scaled = scaler.fit_transform(xTrain)
xVal_scaled = scaler.transform(xVal)
xTest_scaled = scaler.transform(xTest)




# Modelli 
xgModel = XGBRegressor(
    n_estimators = 8000, #5000
    max_depth = 8, #8
    subsample = 0.8, 
    colsample_bytree = 0.8,
    random_state = 42, 
    n_jobs = -1, 
    early_stopping_rounds = 20, 
    learning_rate = 0.01    #0.05
)
xgModel.fit(xTrain_scaled, yTrain, eval_set = [(xVal_scaled, yVal)], verbose = 1000)



lgbModel = LGBMRegressor(
    n_estimators = 15000, #5000
    random_state = 42, 
    learning_rate = 0.05,
    num_leaves = 70, #50
    n_jobs = -1
)
lgbModel.fit(
    xTrain_scaled, yTrain, 
    eval_set = [(xVal_scaled, yVal)], 
    callbacks = [lgb.early_stopping(50)],
    eval_metric = 'rmse'
)




catModel = CatBoostRegressor(
    iterations = 15800, #così è al massimo, posso tentare anche un 13000 oppure un 14000
    learning_rate = 0.1,
    depth = 9, #7
    l2_leaf_reg = 12,
    random_strength = 1,
    loss_function = 'RMSE', 
    eval_metric = 'MAE',
    random_state = 42,
    verbose = 1000 
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
    alpha = 0.1, #0.01                       
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


# Con dataSet Randomizzato
# 6.062307363260181
# 0.9696907164342475
# 67.12707896757912

# Eliminandp il timestamp
# 6.186453824818169
# 0.9688315472125799
# 69.02991246954156


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