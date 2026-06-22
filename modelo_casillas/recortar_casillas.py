from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO


RUTA_PROYECTO = Path(__file__).resolve().parents[1]
RUTA_MODELO = RUTA_PROYECTO / "modelo_yolo/modelos/best.pt"
RUTA_IMAGEN = RUTA_PROYECTO / "modelo_yolo/imag_proc/images/test/sudoku_1008_jpg.rf.WAN6c48NXkKe69G07Ln0.jpg"
CARPETA_SALIDA = RUTA_PROYECTO / "modelo_casillas/casillas_debug"

TAMANO_TABLERO = 450
TAMANO_CASILLA = 50
MARGEN_CASILLA = 5


def cargar_modelo_yolo():
    return YOLO(str(RUTA_MODELO))


def ordenar_puntos(puntos):
    puntos = np.array(puntos, dtype="float32")

    suma = puntos.sum(axis=1)
    diferencia = np.diff(puntos, axis=1).reshape(-1)

    arriba_izquierda = puntos[np.argmin(suma)]
    arriba_derecha = puntos[np.argmin(diferencia)]
    abajo_derecha = puntos[np.argmax(suma)]
    abajo_izquierda = puntos[np.argmax(diferencia)]

    return np.array(
        [arriba_izquierda, arriba_derecha, abajo_derecha, abajo_izquierda],
        dtype="float32",
    )


def recortar_por_cuatro_esquinas(tablero):
    gris = cv2.cvtColor(tablero, cv2.COLOR_BGR2GRAY)
    gris = cv2.GaussianBlur(gris, (5, 5), 0)

    binaria = cv2.adaptiveThreshold(
        gris,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        11,
        2,
    )

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    binaria = cv2.morphologyEx(binaria, cv2.MORPH_CLOSE, kernel, iterations=1)

    contornos, _ = cv2.findContours(
        binaria,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )

    mejor_contorno = None
    mejor_area = 0
    area_imagen = tablero.shape[0] * tablero.shape[1]

    for contorno in contornos:
        area = cv2.contourArea(contorno)

        if area < area_imagen * 0.25:
            continue

        perimetro = cv2.arcLength(contorno, True)
        esquinas = cv2.approxPolyDP(contorno, 0.02 * perimetro, True)

        if len(esquinas) == 4 and area > mejor_area:
            mejor_contorno = esquinas
            mejor_area = area

    if mejor_contorno is None:
        return None

    origen = ordenar_puntos(mejor_contorno.reshape(4, 2))
    destino = np.array(
        [
            [0, 0],
            [TAMANO_TABLERO - 1, 0],
            [TAMANO_TABLERO - 1, TAMANO_TABLERO - 1],
            [0, TAMANO_TABLERO - 1],
        ],
        dtype="float32",
    )

    matriz = cv2.getPerspectiveTransform(origen, destino)
    return cv2.warpPerspective(tablero, matriz, (TAMANO_TABLERO, TAMANO_TABLERO))


def recortar_por_rectangulo_interno(tablero):
    gris = cv2.cvtColor(tablero, cv2.COLOR_BGR2GRAY)
    _, binaria = cv2.threshold(
        gris,
        0,
        255,
        cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU,
    )

    puntos = cv2.findNonZero(binaria)

    if puntos is None:
        return tablero

    x, y, ancho, alto = cv2.boundingRect(puntos)
    area_rectangulo = ancho * alto
    area_imagen = tablero.shape[0] * tablero.shape[1]

    if area_rectangulo < area_imagen * 0.25:
        return tablero

    margen = 2
    x1 = max(0, x - margen)
    y1 = max(0, y - margen)
    x2 = min(tablero.shape[1], x + ancho + margen)
    y2 = min(tablero.shape[0], y + alto + margen)

    return tablero[y1:y2, x1:x2]


def ajustar_recorte_interno(tablero):
    tablero_corregido = recortar_por_cuatro_esquinas(tablero)

    if tablero_corregido is not None:
        return tablero_corregido

    return recortar_por_rectangulo_interno(tablero)


def recortar_tablero(modelo, ruta_imagen, conf=0.25):
    imagen = cv2.imread(str(ruta_imagen))

    if imagen is None:
        raise ValueError(f"No se ha podido leer la imagen: {ruta_imagen}")

    resultados = modelo(str(ruta_imagen), conf=conf, verbose=False)
    resultado = resultados[0]

    if len(resultado.boxes) == 0:
        raise ValueError("No se ha detectado ningun sudoku en la imagen.")

    indice_mejor_caja = int(resultado.boxes.conf.argmax())
    x1, y1, x2, y2 = resultado.boxes.xyxy[indice_mejor_caja].cpu().numpy().astype(int)

    tablero = imagen[y1:y2, x1:x2]
    tablero = ajustar_recorte_interno(tablero)
    tablero = cv2.resize(tablero, (TAMANO_TABLERO, TAMANO_TABLERO))

    return tablero


def dividir_en_casillas(tablero):
    casillas = []

    for fila in range(9):
        for columna in range(9):
            y1 = fila * TAMANO_CASILLA
            y2 = y1 + TAMANO_CASILLA
            x1 = columna * TAMANO_CASILLA
            x2 = x1 + TAMANO_CASILLA

            casilla = tablero[y1:y2, x1:x2]
            casilla = casilla[
                MARGEN_CASILLA : TAMANO_CASILLA - MARGEN_CASILLA,
                MARGEN_CASILLA : TAMANO_CASILLA - MARGEN_CASILLA,
            ]
            casillas.append(casilla)

    return casillas


def guardar_casillas(tablero, casillas, carpeta_salida=CARPETA_SALIDA):
    carpeta_salida.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(carpeta_salida / "tablero_recortado.jpg"), tablero)

    for indice, casilla in enumerate(casillas):
        fila = indice // 9
        columna = indice % 9
        nombre = f"casilla_{fila:02d}_{columna:02d}.jpg"
        cv2.imwrite(str(carpeta_salida / nombre), casilla)


def procesar_imagen(ruta_imagen, carpeta_salida=CARPETA_SALIDA):
    modelo = cargar_modelo_yolo()
    tablero = recortar_tablero(modelo, ruta_imagen)
    casillas = dividir_en_casillas(tablero)
    guardar_casillas(tablero, casillas, carpeta_salida)

    return tablero, casillas


def main():
    procesar_imagen(RUTA_IMAGEN)
    print("Tablero y casillas guardadas correctamente.")
    print(f"Carpeta de salida: {CARPETA_SALIDA}")


if __name__ == "__main__":
    main()
