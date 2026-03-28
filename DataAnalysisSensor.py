import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os

def sanitize_dataset(input_path, output_path):
    """
    Carica il dataset originale, rimuove le righe senza LOAD 
    e salva il risultato in un nuovo file.
    """
    print(f"Sanitizzazione in corso: leggendo {input_path}...")
    df = pd.read_csv(input_path)
    
    # Rimuove le righe dove 'LOAD' è nullo (NaN)
    # In un CSV, le celle vuote (es. ,,) vengono lette come NaN da pandas
    df_clean = df.dropna(subset=['LOAD'])
    
    # Salva il file sanitificato
    df_clean.to_csv(output_path, index=False)
    print(f"Dataset sanitificato salvato in: {output_path}")
    print(f"Righe rimosse: {len(df) - len(df_clean)}")
    return df_clean

def perform_analysis(df, sensor_cols):
    #Esegue l'analisi statistica e genera i grafici sul dataframe fornito.

    print("\n--- Analisi dei dati sanitificati ---")
    
    # 1. Temperature medie per ogni sensore
    sensor_means = df[sensor_cols].mean()
    print("\nTemperature medie per ogni sensore:")
    print(sensor_means)

    # 2. Temperatura media generale
    overall_mean = df[sensor_cols].values.mean()
    print(f"\nTemperatura media generale: {overall_mean:.2f}")

    # 3. Temperature medie per fascia oraria (TIMESTAMP)
    hourly_means = df.groupby('TIMESTAMP')[sensor_cols].mean().mean(axis=1)
    print("\nTemperature medie per fascia oraria:")
    print(hourly_means)

    # 4. Matrice di varianza (Varianza per ogni sensore)
    var_matrix = df[sensor_cols].var()
    print("\nVarianza per ogni sensore:")
    print(var_matrix)

    # 5. Matrice di covarianza e correlazione
    cov_matrix = df[sensor_cols].cov()
    corr_matrix = df[sensor_cols].corr()
    
    # 6. Calcolo del Rango della Matrice(ATTENZIONE INUTILE PERCHÉ É OVVIO SIA MASSIMO)
    # Usiamo la matrice dei dati originale per Completezza anche se inutile
    matrix_data = df[sensor_cols].values
    rank = np.linalg.matrix_rank(matrix_data)
    print(f"\n--- Analisi del Rango ---")
    print(f"Rango della matrice dei sensori (25 colonne): {rank}")
    if rank == len(sensor_cols):
        print("Il rango è MASSIMO (Full Rank). Le colonne sono linearmente indipendenti.")
    else:
        print(f"Il rango è ridotto ({rank}/25). Esistono dipendenze lineari esatte.")
    
    # --- Generazione Grafici ---
    print("\nGenerazione dei grafici...")
    sns.set_theme(style="whitegrid")

    # Figura 1: Distribuzione Globale
    plt.figure(figsize=(10, 6))
    sns.histplot(df[sensor_cols].values.flatten(), bins=50, kde=True, color='skyblue')
    plt.title('Distribuzione Globale delle Temperature (Dati Sanitificati)')
    plt.xlabel('Temperatura')
    plt.ylabel('Frequenza')
    plt.savefig('./Grafici/DataAnalysis/distribuzione_globale.png')
    plt.close()

    # Figura 2: Heatmap Correlazione
    plt.figure(figsize=(14, 12))
    # Utilizzo YlOrRd per un contrasto migliore e aggiungo i valori numerici
    sns.heatmap(corr_matrix, 
                annot=True, 
                fmt=".2f", 
                cmap='YlOrRd', 
                linewidths=.5, 
                annot_kws={"size": 8},
                cbar_kws={'label': 'Grado di Correlazione (0-1)'})
    plt.title('Matrice di Correlazione tra i Sensori (Dettagliata)')
    plt.savefig('./Grafici/DataAnalysis/matrice_correlazione.png')
    plt.close()

    # Figura 3: Media Oraria
    plt.figure(figsize=(10, 6))
    hourly_means.plot(kind='line', marker='o', color='red')
    plt.title('Andamento Temperatura Media per Fascia Oraria')
    plt.xlabel('Ora del Giorno (TIMESTAMP)')
    plt.ylabel('Temperatura Media')
    plt.xticks(range(0, 24))
    plt.grid(True)
    plt.savefig('./Grafici/DataAnalysis/media_oraria.png')
    plt.close()

    # Figura 4: Boxplot Sensori
    plt.figure(figsize=(15, 8))
    sns.boxplot(data=df[sensor_cols])
    plt.title('Distribuzione delle Temperature per Singolo Sensore (Sanitificato)')
    plt.xlabel('Sensore')
    plt.ylabel('Temperatura')
    plt.savefig('./Grafici/DataAnalysis/boxplot_sensori.png')
    plt.close()

    print("\nAnalisi completata. I grafici e il file CSV sanitificato sono pronti.")

if __name__ == "__main__":
    input_file = 'L1_train.csv'
    output_file = 'Dataset_trainingSanitificato.csv'
    
    # Definizione colonne sensori
    sensors = [f'w{i}' for i in range(1, 26)]
    
    if os.path.exists(input_file):
        # Fase 1: Sanitizzazione
        df_sanitizzato = sanitize_dataset(input_file, output_file)
        
        # Fase 2: Analisi sul dataset pulito
        perform_analysis(df_sanitizzato, sensors)
    else:
        print(f"Errore: File {input_file} non trovato nella directory corrente.")
