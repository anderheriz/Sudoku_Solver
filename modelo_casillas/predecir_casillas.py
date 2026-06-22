from pathlib import Path
import sys

import cv2
import numpy as np
import tensorflow as tf

from preprocesar_casilla import limpiar_casilla


RUTA_PROYECTO = Path(__file__).resolve().parents[1]
CARPETA_CASILLAS = RUTA_PROYECTO / "modelo_casillas/casillas_debug"
RUTA_MODELO = RUTA_PROYECTO / "modelo_casillas/modelos/cnn_casillas.keras"
UMBRAL_CONFIANZA_DIGITO = 0.75
UMBRAL_PIXELES_DIGITO = 0.015

sys.path.append(str(RUTA_PROYECTO / "modelo_juego"))
from predecir_solucion import predecir_solucion


def cargar_modelo_cnn():
    return tf.keras.models.load_model(RUTA_MODELO)


def preparar_casilla_limpia(casilla):
    if isinstance(casilla, (str, Path)):
        imagen = cv2.imread(str(casilla), cv2.IMREAD_GRAYSCALE)
    else:
        imagen = casilla

        if len(imagen.shape) == 3:
            imagen = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)

    if imagen is None:
        raise ValueError("No se ha podido leer una casilla.")

    return limpiar_casilla(imagen)


def preparar_imagen(casilla):
    imagen = preparar_casilla_limpia(casilla)
    return imagen.reshape(1, 50, 50, 1)


def parece_vacia(imagen_limpia):
    proporcion_tinta = np.mean(imagen_limpia > 0)
    return proporcion_tinta < UMBRAL_PIXELES_DIGITO


def predecir_casilla(modelo, casilla):
    imagen_limpia = preparar_casilla_limpia(casilla)

    if parece_vacia(imagen_limpia):
        return 0, 1.0

    prediccion = modelo.predict(imagen_limpia.reshape(1, 50, 50, 1), verbose=False)[0]
    clase = int(np.argmax(prediccion))
    confianza = float(np.max(prediccion))

    if clase != 0 and confianza < UMBRAL_CONFIANZA_DIGITO:
        clase = 0

    return clase, confianza


def predecir_tablero(casillas=None, carpeta_casillas=CARPETA_CASILLAS, modelo=None):
    if modelo is None:
        modelo = cargar_modelo_cnn()
        
    tablero = []
    confianzas = []

    for fila in range(9):
        fila_predicha = []
        fila_confianza = []

        for columna in range(9):
            indice = fila * 9 + columna

            if casillas is None:
                casilla = carpeta_casillas / f"casilla_{fila:02d}_{columna:02d}.jpg"
            else:
                casilla = casillas[indice]

            clase, confianza = predecir_casilla(modelo, casilla)

            fila_predicha.append(clase)
            fila_confianza.append(round(confianza, 3))

        tablero.append(fila_predicha)
        confianzas.append(fila_confianza)

    return tablero, confianzas


def resolver_tablero(tablero, modelo_solucion=None):
    return predecir_solucion(tablero, modelo=modelo_solucion)


def imprimir_matriz(matriz):
    for fila in matriz:
        print(fila)


def main():
    tablero, confianzas = predecir_tablero()

    print("Tablero predicho por la CNN:")
    imprimir_matriz(tablero)

    print("\nConfianzas:")
    imprimir_matriz(confianzas)

    print("\nNota: 0 significa casilla vacia antes de resolver.")

    tablero_resuelto = resolver_tablero(tablero)

    if tablero_resuelto is not None:
        print("\nSudoku resuelto:")
        imprimir_matriz(tablero_resuelto)
    else:
        print("\nNo se ha podido resolver el sudoku predicho.")


if __name__ == "__main__":
    main()
