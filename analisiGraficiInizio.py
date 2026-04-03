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
datiMedia = ds.iloc[35066:, 2:27].mean(axis = 1)

dataGrafici = pd.DataFrame()
dataGrafici['ora'] = ds.iloc[35066:, 0]
dataGrafici['consumo'] = consumo
dataGrafici['mediaT'] = datiMedia


# --- grafico 3D ---

fig = px.scatter_3d(
    dataGrafici, 
    x='mediaT',    
    y='ora',       
    z='consumo',              
    
    # Dimensione dei punti basata sul consumo
    size='consumo', 
    size_max=10,              
    
    opacity=0.6,             
    
    title='Analisi 3D: Consumo vs Temperatura e Ora',
    labels={
        'temperatura_media': 'Temp (°C)',
        'ora_del_giorno': 'Ora (0-23)',
        'Consumo': 'Consumo (kW)'
    }
)


fig.update_layout(
    scene = dict(
        xaxis = dict(backgroundcolor="rgba(200, 200, 200, 0.1)"),
        yaxis = dict(backgroundcolor="rgba(200, 200, 200, 0.1)"),
        zaxis = dict(backgroundcolor="rgba(200, 200, 200, 0.1)")
    ),
    margin=dict(r=20, l=10, b=10, t=40)
)


fig.show()


