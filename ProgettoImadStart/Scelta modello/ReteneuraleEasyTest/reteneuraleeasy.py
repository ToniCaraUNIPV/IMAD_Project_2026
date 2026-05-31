import os
import sys
import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from keras import layers, models

#low code platform



ds = pd.read_csv('Dataset/Dataset_random.csv')
consumo = ds['LOAD']
dati = ds.drop(columns = ['LOAD'])

dati['oraSin'] = np.sin(dati['TIMESTAMP'] * (2 * np.pi / 24))
dati['oraCos'] = np.cos(dati['TIMESTAMP'] * (2 * np.pi / 24))
dati = dati.drop(columns = ['TIMESTAMP'])


xTrain, xTest, yTrain, yTest = train_test_split(dati, consumo, test_size = 0.15, random_state = 42)




# Scalo le varie colonne
scaler = StandardScaler()
xTrain_scaled = scaler.fit_transform(xTrain)
xTest_scaled = scaler.transform(xTest)

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

predictRete = mpl.predict(xTest_scaled)

print(mean_absolute_error(predictRete, yTest))
print(r2_score(yTest, predictRete))
print(mean_squared_error(predictRete, yTest))