from pathlib import Path
import random

import cv2
import numpy as np


RUTA_PROYECTO = Path(__file__).resolve().parents[1]
CARPETA_DATASET = RUTA_PROYECTO / "modelo_casillas/dataset_real_aumentado"

AUMENTOS_POR_CASILLA = 40


TABLERO_EJEMPLO_2 = [
    [1, 0, 6, 0, 0, 9, 0, 0, 0],
    [0, 2, 0, 0, 0, 8, 0, 6, 0],
    [0, 4, 3, 0, 0, 1, 0, 0, 0],
    [9, 0, 0, 0, 7, 0, 0, 0, 1],
    [0, 8, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 2, 0, 0, 3, 4, 0],
    [0, 0, 0, 0, 9, 0, 0, 0, 7],
    [6, 0, 0, 0, 0, 0, 0, 1, 0],
    [0, 9, 0, 0, 4, 0, 8, 0, 0],
]

TABLERO_EJEMPLO_3 = [
    [0, 7, 0, 8, 0, 9, 0, 1, 0],
    [3, 0, 9, 6, 0, 0, 0, 0, 5],
    [0, 6, 1, 2, 4, 0, 7, 0, 0],
    [0, 3, 0, 0, 6, 1, 5, 0, 0],
    [9, 5, 0, 3, 2, 4, 0, 0, 8],
    [1, 2, 0, 0, 0, 0, 6, 0, 0],
    [6, 0, 5, 0, 0, 0, 2, 3, 0],
    [2, 0, 0, 0, 0, 0, 0, 0, 7],
    [0, 1, 8, 5, 0, 0, 0, 6, 4],
]

TABLERO_EJEMPLO_4 = [
    [0, 0, 0, 0, 0, 0, 6, 0, 0],
    [0, 0, 0, 5, 6, 0, 0, 2, 1],
    [6, 8, 0, 2, 0, 0, 5, 0, 7],
    [0, 6, 0, 0, 0, 7, 0, 0, 0],
    [0, 0, 8, 0, 1, 0, 3, 0, 0],
    [0, 0, 0, 3, 0, 0, 0, 5, 0],
    [9, 0, 5, 0, 0, 1, 0, 3, 6],
    [8, 4, 0, 0, 3, 5, 0, 0, 0],
    [0, 0, 2, 0, 0, 0, 0, 0, 0],
]

TABLERO_EJEMPLO_5 = [
    [0, 0, 9, 5, 8, 6, 0, 0, 0],
    [0, 0, 0, 0, 2, 0, 0, 0, 0],
    [4, 0, 0, 0, 0, 0, 6, 8, 3],
    [9, 0, 0, 6, 5, 0, 0, 3, 2],
    [0, 6, 0, 7, 0, 0, 0, 9, 8],
    [0, 3, 0, 2, 0, 0, 7, 0, 4],
    [0, 0, 3, 0, 0, 0, 0, 0, 0],
    [6, 2, 0, 0, 1, 5, 0, 4, 0],
    [0, 0, 0, 4, 0, 0, 0, 5, 0],
]

TABLERO_EJEMPLO_6 = [
    [0, 9, 0, 5, 0, 6, 0, 8, 0],
    [0, 1, 3, 0, 9, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 7, 0, 0, 0],
    [0, 0, 0, 0, 4, 0, 0, 0, 0],
    [0, 6, 9, 0, 5, 0, 0, 1, 0],
    [0, 0, 7, 3, 0, 0, 0, 0, 9],
    [0, 0, 4, 6, 0, 0, 0, 0, 3],
    [5, 0, 0, 0, 0, 4, 0, 6, 0],
    [2, 0, 0, 0, 0, 0, 8, 0, 0],
]

EJEMPLOS_ENTRENAMIENTO = [
    {
        "nombre": "sudoku_2",
        "carpeta": RUTA_PROYECTO / "modelo_casillas/casillas_entrenamiento/sudoku_2",
        "tablero": TABLERO_EJEMPLO_2,
    },
    {
        "nombre": "sudoku_3",
        "carpeta": RUTA_PROYECTO / "modelo_casillas/casillas_entrenamiento/sudoku_3",
        "tablero": TABLERO_EJEMPLO_3,
    },
    {
        "nombre": "sudoku_azul",
        "carpeta": RUTA_PROYECTO / "modelo_casillas/casillas_entrenamiento/sudoku_azul",
        "tablero": TABLERO_EJEMPLO_4,
    },
    {
        "nombre": "sudoku_online",
        "carpeta": RUTA_PROYECTO / "modelo_casillas/casillas_entrenamiento/sudoku_online",
        "tablero": TABLERO_EJEMPLO_5,
    },
    {
        "nombre": "sudoku_b_n",
        "carpeta": RUTA_PROYECTO / "modelo_casillas/casillas_entrenamiento/sudoku_b_n",
        "tablero": TABLERO_EJEMPLO_6,
    },
]


def aumentar_imagen(imagen):
    alto, ancho = imagen.shape[:2]

    brillo = random.randint(-20, 20)
    imagen = np.clip(imagen.astype(np.int16) + brillo, 0, 255).astype(np.uint8)

    angulo = random.uniform(-4, 4)
    escala = random.uniform(0.95, 1.05)
    centro = (ancho // 2, alto // 2)
    matriz = cv2.getRotationMatrix2D(centro, angulo, escala)

    matriz[0, 2] += random.randint(-2, 2)
    matriz[1, 2] += random.randint(-2, 2)

    imagen = cv2.warpAffine(imagen, matriz, (ancho, alto), borderValue=235)

    if random.random() < 0.3:
        imagen = cv2.GaussianBlur(imagen, (3, 3), 0)

    return imagen


def guardar_imagen(imagen, etiqueta, contador):
    split = "train" if contador % 5 != 0 else "val"
    carpeta = CARPETA_DATASET / split / str(etiqueta)
    carpeta.mkdir(parents=True, exist_ok=True)

    ruta = carpeta / f"{etiqueta}_{contador:05d}.jpg"
    cv2.imwrite(str(ruta), imagen)


def main():
    random.seed(42)
    contador = 0

    for ejemplo in EJEMPLOS_ENTRENAMIENTO:
        carpeta_casillas = ejemplo["carpeta"]
        tablero_etiquetado = ejemplo["tablero"]

        for fila in range(9):
            for columna in range(9):
                etiqueta = tablero_etiquetado[fila][columna]
                ruta_casilla = carpeta_casillas / f"casilla_{fila:02d}_{columna:02d}.jpg"
                imagen = cv2.imread(str(ruta_casilla), cv2.IMREAD_GRAYSCALE)

                if imagen is None:
                    raise ValueError(f"No se ha podido leer: {ruta_casilla}")

                for _ in range(AUMENTOS_POR_CASILLA):
                    imagen_aumentada = aumentar_imagen(imagen)
                    guardar_imagen(imagen_aumentada, etiqueta, contador)
                    contador += 1

    print("Dataset real aumentado creado correctamente.")
    print(f"Ejemplos usados: {len(EJEMPLOS_ENTRENAMIENTO)}")
    print(f"Imagenes generadas: {contador}")
    print(f"Carpeta: {CARPETA_DATASET}")


if __name__ == "__main__":
    main()
