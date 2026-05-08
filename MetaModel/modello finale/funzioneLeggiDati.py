import pandas as pd
import joblib
import numpy as np
import re
import warnings

# Disattiva i warning del LightGBM per pulire la console
warnings.filterwarnings("ignore", category=UserWarning)


def predict_modello(nome_file_input, nome_file_output):
    try:

        with open(nome_file_input, 'r') as f:
            linee = f.readlines()

        dati_puliti = []
        for riga in linee:
            numeri = re.findall(r"[-+]?\d*\.\d+|\d+", riga)
            
            if len(numeri) >= 26:
                dati_puliti.append([float(n) for n in numeri[:26]])

        if not dati_puliti:
            print("Nessun dato valido trovato nel file!")
            return

        
        df = pd.DataFrame(dati_puliti)
        
        
        m = joblib.load('modello_finale.joblib')
        
        
        colonne_base = ['timestamp'] + [f'w{i}' for i in range(1, 26)]
        df.columns = colonne_base

       
        df['oraSin'] = np.sin(df['timestamp'] * (2 * np.pi / 24))
        df['oraCos'] = np.cos(df['timestamp'] * (2 * np.pi / 24))

        
        X = df[m['scaler'].feature_names_in_]

        
        X_scaled = m['scaler'].transform(X)
        
        preds = pd.DataFrame({
            'xgb': m['xgb'].predict(X_scaled),
            'lgb': m['lgbm'].predict(X_scaled),
            'cat': m['cat'].predict(X_scaled),
            'rete': m['rete'].predict(X_scaled)
        })

        pred_finale = m['meta_model'].predict(preds)
        
       
        output = pd.DataFrame({
            'Timestamp': df['timestamp'],
            'Consumo_Previsto': pred_finale
        })
        output.to_csv(nome_file_output, sep=';', index=False)
        
        print(f"Elaborazione completata con successo!")
        print(f"File generato: {nome_file_output} ({len(df)} righe elaborate)")

    except Exception as e:
        print(f"Errore imprevisto: {e}")