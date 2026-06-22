from pathlib import Path
import sys

import cv2
import numpy as np


RUTA_PROYECTO = Path(__file__).resolve().parents[1]
CARPETA_SALIDA = RUTA_PROYECTO / "modelo_casillas/debug_ultima_imagen"


def guardar_contact_sheet(casillas, ruta_salida, titulo_predicciones=None):
    alto_celda = 70
    ancho_celda = 70
    lienzo = np.full((alto_celda * 9, ancho_celda * 9, 3), 255, dtype=np.uint8)

    for indice, casilla in enumerate(casillas):
        fila = indice // 9
        columna = indice % 9

        if len(casilla.shape) == 2:
            casilla_color = cv2.cvtColor(casilla, cv2.COLOR_GRAY2BGR)
        else:
            casilla_color = casilla.copy()

        casilla_color = cv2.resize(casilla_color, (50, 50))
        y = fila * alto_celda
        x = columna * ancho_celda
        lienzo[y : y + 50, x : x + 50] = casilla_color

        if titulo_predicciones is not None:
            texto = titulo_predicciones[fila][columna]
            cv2.putText(
                lienzo,
                texto,
                (x + 5, y + 65),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (0, 0, 180),
                1,
                cv2.LINE_AA,
            )

    cv2.imwrite(str(ruta_salida), lienzo)


def guardar_matriz(nombre, matriz):
    ruta = CARPETA_SALIDA / nombre

    with open(ruta, "w", encoding="utf-8") as archivo:
        for fila in matriz:
            archivo.write(",".join(str(valor) for valor in fila))
            archivo.write("\n")


def diagnosticar(ruta_imagen):
    from recortar_casillas import cargar_modelo_yolo, dividir_en_casillas, recortar_tablero
    from predecir_casillas import (
        cargar_modelo_cnn,
        predecir_casilla,
        preparar_casilla_limpia,
    )

    CARPETA_SALIDA.mkdir(parents=True, exist_ok=True)

    modelo_yolo = cargar_modelo_yolo()
    modelo_cnn = cargar_modelo_cnn()

    tablero = recortar_tablero(modelo_yolo, ruta_imagen)
    casillas = dividir_en_casillas(tablero)
    casillas_limpias = [preparar_casilla_limpia(casilla) for casilla in casillas]

    tablero_predicho = []
    confianzas = []
    textos_contact_sheet = []

    for fila in range(9):
        fila_predicha = []
        fila_confianza = []
        fila_texto = []

        for columna in range(9):
            indice = fila * 9 + columna
            clase, confianza = predecir_casilla(modelo_cnn, casillas[indice])
            fila_predicha.append(clase)
            fila_confianza.append(round(confianza, 3))
            fila_texto.append(f"{clase} {confianza:.2f}")

        tablero_predicho.append(fila_predicha)
        confianzas.append(fila_confianza)
        textos_contact_sheet.append(fila_texto)

    cv2.imwrite(str(CARPETA_SALIDA / "tablero_detectado.jpg"), tablero)
    guardar_contact_sheet(casillas, CARPETA_SALIDA / "casillas_originales.jpg")
    guardar_contact_sheet(
        casillas_limpias,
        CARPETA_SALIDA / "casillas_limpias_predicciones.jpg",
        textos_contact_sheet,
    )
    guardar_matriz("tablero_predicho.csv", tablero_predicho)
    guardar_matriz("confianzas.csv", confianzas)

    print("Diagnostico guardado correctamente.")
    print(f"Carpeta: {CARPETA_SALIDA}")
    print("Tablero predicho:")
    for fila in tablero_predicho:
        print(fila)


def main():
    if len(sys.argv) < 2:
        print("Uso: python diagnosticar_imagen.py ruta/a/imagen.jpg")
        return

    diagnosticar(Path(sys.argv[1]))


if __name__ == "__main__":
    main()
