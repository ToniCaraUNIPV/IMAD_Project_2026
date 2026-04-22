from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
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

#9.758335353198568
#0.9276303339690215
#158.74509195086074

# Con (256, 128, 64, 32)
#9.753318398546629
#0.927760137212634
#158.4603645371133
#Ma è molto più lento




# Grafico
ore = 200

plt.figure(figsize = (12, 7))

plt.plot(range(ore), yTest[:ore], label = 'modello reale', color = 'blue')
plt.plot(range(ore), predictModel[:ore], label = 'modello Rete', color = 'red', linestyle = '--')
plt.xlabel('ore')
plt.ylabel('consumo')
plt.title('Confronto modello')
plt.legend()

plt.show()