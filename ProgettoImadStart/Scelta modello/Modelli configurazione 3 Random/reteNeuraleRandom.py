from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
import matplotlib.pyplot as plt




ds = pd.read_csv('Dataset_random.csv')
consumo = ds['LOAD']
dati = ds.drop(columns = ['LOAD'])

dati['oraSin'] = np.sin(dati.iloc[:, 0] * (2 * np.pi / 24))
dati['oraCos'] = np.cos(dati.iloc[:, 0] * (2 * np.pi / 24))



xTrain, xTest, yTrain, yTest = train_test_split(dati, consumo, test_size = 0.15, random_state = 42)



# Scalo i vari dati
scaler = StandardScaler()
xTrain_scaled = scaler.fit_transform(xTrain)
xTest_scaled = scaler.transform(xTest)


# Rete Neurale
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


# Prestazioni rete
predictModel = mpl.predict(xTest_scaled)


print(mean_absolute_error(yTest, predictModel))
print(r2_score(yTest, predictModel))
print(mean_squared_error(yTest, predictModel))


# Con dataSet randomizzato
# 7.8810578474482575
# 0.9519416759751103
# 106.43652809772998




# Grafico
ore = 7 * 24

plt.figure(figsize = (12, 7))

plt.plot(range(ore), yTest[:ore], label = 'modello reale', color = 'blue')
plt.plot(range(ore), predictModel[:ore], label = 'modello Rete', color = 'red', linestyle = '--')
plt.xlabel('ore')
plt.ylabel('consumo')
plt.title('Confronto modello')
plt.legend()

plt.show()