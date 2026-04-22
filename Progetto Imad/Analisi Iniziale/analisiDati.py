from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import Ridge
from sklearn.ensemble import StackingRegressor
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
from xgboost import XGBRegressor
from sklearn.model_selection import RandomizedSearchCV
from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor
import lightgbm as lgb
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px





ds = pd.read_csv('L1_train.csv')
consumo = ds.iloc[35066:, 1]
dati = ds.iloc[35066:, np.r_[0, 2:27]]

dati['oraSin'] = np.sin(dati.iloc[:, 0] * (2 * np.pi / 24))
dati['oraCos'] = np.cos(dati.iloc[:, 0] * (2 * np.pi / 24))

xTemp, xTest, yTemp, yTest = train_test_split(dati, consumo, test_size = 0.15, random_state = 42)
xTrain, xVal, yTrain, yVal = train_test_split(xTemp, yTemp, test_size = 0.18, random_state  = 42)


#Standardizzo
std = StandardScaler()

x_train_std = std.fit_transform(xTrain)
x_val_std = std.transform(xVal)
x_test_std = std.transform(xTest)


pca = PCA(n_components = 0.95)
sensoriRidotti = pca.fit_transform(x_train_std[:, 2:26])

print(f"passo da 25 sensori a {pca.n_components_} sensori")
#print(sensoriRidotti)