# Progetto IMAD 2026 - Load Prediction
**Università degli Studi di Pavia**  
**Corso:** Identificazione Modelli e Analisi dei Dati (IMAD)

Questo progetto è stato sviluppato per il corso di **IMAD**. L'obiettivo principale è la previsione del carico energetico (`LOAD`) basata su dati temporali e letture provenienti da 25 sensori ambientali.

## 📋 Descrizione del Progetto

Il sistema analizza una serie storica di dati di consumo energetico correlandoli con 25 sensori differenti. Attraverso tecniche di Machine Learning e Deep Learning, è stato sviluppato un modello in grado di prevedere con alta precisione il consumo futuro.

Per una panoramica completa dei risultati e della metodologia, consultare il file: `Presentazione IMAD.pdf`.

Il progetto include:
- Analisi esplorativa dei dati (EDA) e visualizzazione.
- Preprocessing avanzato (trasformazioni cicliche, scaling, PCA).
- Implementazione di modelli basati su alberi di decisione (XGBoost, LightGBM, CatBoost).
- Sviluppo di Reti Neurali (MLP e CNN-LSTM per serie temporali).
- Un modello finale di **Blending (Meta-Model)** che combina i migliori predittori.

## 📊 Dataset

Il dataset principale (`L1_train.csv`) è composto dalle seguenti colonne:
- `TIMESTAMP`: Ora del rilevamento (0-23).
- `LOAD`: Variabile target (Consumo energetico).
- `w1` - `w25`: Letture dei sensori ambientali.

## 🛠️ Requisiti Tecnici

Per eseguire il progetto, è necessario avere installato Python (consigliato 3.10+) e le seguenti librerie:
- `pandas`
- `numpy`
- `scikit-learn`
- `xgboost`
- `lightgbm`
- `catboost`
- `joblib`
- `matplotlib` / `seaborn` (per la generazione dei grafici)

## 🤖 Modelli Implementati

1. **Gradient Boosting**: Implementazioni con XGBoost, LightGBM e CatBoost con tuning dei iperparametri.
2. **Reti Neurali (Deep Learning)**:
   - **MLP (Multi-Layer Perceptron)**: Architetture dense testate con diverse funzioni di perdita (MSE, Huber).
   - **CNN-LSTM**: Modelli avanzati per catturare dipendenze spaziali e temporali nei dati.
3. **Meta-Modello (Blending)**: Un regressore finale (**RidgeCV**) che combina le previsioni di XGB, LGBM, CatBoost e della Rete Neurale.

## 📂 Struttura del Repository

- `Dataset/`: Contiene i file CSV utilizzati per il training e il testing.
- `Grafici/`: Raccolta di grafici generati durante l'analisi.
- `Modelli/`:
  - `Alberi/`: Script per i modelli basati su gradient boosting.
  - `ReteNeuraleModel/`: Implementazioni di reti neurali dense.
  - `ReteNeurale_CNN/`: Modelli ibridi CNN-LSTM.
  - `MetaModel/modello finale/`: **Versione di produzione** del modello di blending.
- `ProgettoImadStart/`: Script iniziali di analisi e test dei primi modelli.
- `Presentazione IMAD.pdf`: Documentazione dettagliata del progetto e dei risultati.

## 🚀 Come Utilizzare il Modello Finale

Per eseguire previsioni utilizzando il meta-modello finale:

1. Navigare nella cartella `Modelli/MetaModel/modello finale/`.
2. Inserire i dati di input nel file `input.txt`. Il formato deve essere: `timestamp, w1, w2, ..., w25`.
3. Eseguire lo script `main.py`.
4. Le previsioni verranno generate nel file `previsioni.txt`.

---
*Progetto realizzato per l'Università degli Studi di Pavia - Anno 2026.*
