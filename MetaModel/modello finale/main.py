from funzioneLeggiDati import predict_modello
import os



# ISTRUZIONI
"Il file input.txt deve contenere come primo valore il timestamp seguito dai 25 sensori, il separatore può essere la virgola"
"Il file input.txt deve trovarsi nella stessa cartella in cui si lancia il programma main.py"
#

print(f"Python sta cercando i file in questa cartella: {os.getcwd()}")

predict_modello("input.txt", "previsioni.txt")



