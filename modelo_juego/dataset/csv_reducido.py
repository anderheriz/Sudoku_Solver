from pathlib import Path

import pandas as pd


CARPETA_DATASET = Path(__file__).resolve().parent
RUTA_ENTRADA = CARPETA_DATASET / "sudoku.csv"
RUTA_SALIDA = CARPETA_DATASET / "sudoku_1M.csv"
N_FILAS = 1_000_000


df = pd.read_csv(RUTA_ENTRADA, dtype=str)
df.head(N_FILAS).to_csv(RUTA_SALIDA, index=False)

print("CSV reducido creado:")
print(RUTA_SALIDA)
print(df.head(N_FILAS).shape)
