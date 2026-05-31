import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import RidgeCV
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
from xgboost import XGBRegressor
from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor
import lightgbm as lgb
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from funzioneGraficiAdattata import evaluate_metrics, plot_results
import joblib

#low code platform



ds = pd.read_csv('Dataset_random.csv')
consumo = ds[['LOAD']]
dati = ds.drop(columns = ['LOAD'])

dati['oraSin'] = np.sin(dati['TIMESTAMP'] * (2 * np.pi / 24))
dati['oraCos'] = np.cos(dati['TIMESTAMP'] * (2 * np.pi / 24))
dati = dati.drop(columns = ['TIMESTAMP'])



xTemp, xTest, yTemp, yTest = train_test_split(dati, consumo, test_size = 0.15, random_state = 42)
xTrain, xVal, yTrain, yVal = train_test_split(xTemp, yTemp, test_size = 0.18, random_state = 42)




# Scalo le varie colonne
scaler = StandardScaler()

xTrain_scaled = scaler.fit_transform(xTrain)
xVal_scaled = scaler.transform(xVal)
xTest_scaled = scaler.transform(xTest)




# Modelli 
xgModel = XGBRegressor(
    n_estimators = 5000, #5000
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
    n_estimators = 5000, #5000
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
    iterations = 8000, #così è al massimo, posso tentare anche un 13000 oppure un 14000
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

metaModel = RidgeCV(cv = 5)
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
# 6.038079863391565
# 0.9698868340611235
# 66.69273008573623



predictVal = metaModel.predict(nuovoDataSet)


yVal_reale = yVal.values.flatten()
ypVal_reale = predictVal # Le predizioni ora escono già in formato "consumo reale"

yTest_reale = yTest.values.flatten()
ypTest_reale = predictFinale

met_val = evaluate_metrics(yVal_reale, ypVal_reale, label="Validazione")
met_test = evaluate_metrics(yTest_reale, ypTest_reale, label="Test")


# Generiamo il mega-grafico
plot_results(
    y_val=yVal_reale, yp_val=ypVal_reale,
    y_test=yTest_reale, yp_test=ypTest_reale,
    metrics_val=met_val, metrics_test=met_test,
    save_path="report_finale_blending.png",
    titolo_modello = "blendig_metaModel"
)





# Creo file .joblib

modello_caratteristiche = {
    'scaler': scaler,
    'xgb': xgModel,
    'lgbm': lgbModel,
    'cat': catModel,
    'rete': mpl,
    'meta_model': metaModel
}

joblib.dump(modello_caratteristiche, 'modello_finale.joblib')