from pathlib import Path

import numpy as np
import pandas as pd


RUTA_DATASET = Path(__file__).resolve().parent / "dataset/sudoku.csv"


def texto_a_tablero(texto):
    texto = str(texto).zfill(81)
    valores = [int(numero) for numero in texto]
    tablero = []

    for fila in range(9):
        inicio = fila * 9
        fin = inicio + 9
        tablero.append(valores[inicio:fin])

    return tablero


def tablero_a_texto(tablero):
    numeros = []

    for fila in tablero:
        for numero in fila:
            numeros.append(str(numero))

    return "".join(numeros)


def texto_a_vector(texto):
    texto = str(texto).zfill(81)
    return np.array([int(numero) for numero in texto], dtype=np.int64)


def tablero_a_vector(tablero):
    return np.array([numero for fila in tablero for numero in fila], dtype=np.int64)


def cargar_ejemplos(n=5):
    ejemplos = pd.read_csv(RUTA_DATASET, nrows=n, dtype=str)
    return ejemplos


def cargar_dataset(n_filas=None):
    return pd.read_csv(RUTA_DATASET, nrows=n_filas, dtype=str)


def preparar_datos(n_filas=None):
    datos = cargar_dataset(n_filas)

    x = np.array([texto_a_vector(texto) for texto in datos["puzzle"]], dtype=np.int64)
    y = np.array([texto_a_vector(texto) for texto in datos["solution"]], dtype=np.int64)

    # La solucion solo tiene numeros 1-9. Para entrenar con 9 clases:
    # 1 -> clase 0, 2 -> clase 1, ..., 9 -> clase 8.
    y = y - 1

    return x, y


def cargar_ejemplo(indice=0):
    ejemplos = pd.read_csv(RUTA_DATASET, skiprows=range(1, indice + 1), nrows=1, dtype=str)
    fila = ejemplos.iloc[0]

    puzzle = texto_a_tablero(fila["puzzle"])
    solution = texto_a_tablero(fila["solution"])

    return puzzle, solution


def main():
    ejemplos = cargar_ejemplos(3)
    print(ejemplos)


if __name__ == "__main__":
    main()
