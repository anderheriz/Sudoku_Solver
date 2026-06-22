import cv2
import numpy as np


TAMANO_IMAGEN = 50


def convertir_a_gris(imagen):
    if len(imagen.shape) == 3:
        return cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)

    return imagen


def quitar_bordes(imagen_binaria, margen=4):
    imagen = imagen_binaria.copy()
    imagen[:margen, :] = 0
    imagen[-margen:, :] = 0
    imagen[:, :margen] = 0
    imagen[:, -margen:] = 0
    return imagen


def quitar_lineas_de_borde(imagen_binaria):
    numero_etiquetas, etiquetas, estadisticas, _ = cv2.connectedComponentsWithStats(imagen_binaria)
    salida = imagen_binaria.copy()

    for etiqueta in range(1, numero_etiquetas):
        x = estadisticas[etiqueta, cv2.CC_STAT_LEFT]
        y = estadisticas[etiqueta, cv2.CC_STAT_TOP]
        ancho = estadisticas[etiqueta, cv2.CC_STAT_WIDTH]
        alto = estadisticas[etiqueta, cv2.CC_STAT_HEIGHT]

        linea_horizontal = alto <= 3 and ancho >= 12
        linea_vertical = ancho <= 3 and alto >= 12
        cerca_borde_horizontal = y < 7 or y + alto > TAMANO_IMAGEN - 7
        cerca_borde_vertical = x < 7 or x + ancho > TAMANO_IMAGEN - 7

        if (linea_horizontal and cerca_borde_horizontal) or (linea_vertical and cerca_borde_vertical):
            salida[etiquetas == etiqueta] = 0

    return salida


def centrar_digito(imagen_binaria):
    puntos = cv2.findNonZero(imagen_binaria)

    if puntos is None:
        return np.zeros((TAMANO_IMAGEN, TAMANO_IMAGEN), dtype=np.uint8)

    x, y, ancho, alto = cv2.boundingRect(puntos)

    if ancho < 3 or alto < 3:
        return np.zeros((TAMANO_IMAGEN, TAMANO_IMAGEN), dtype=np.uint8)

    digito = imagen_binaria[y : y + alto, x : x + ancho]

    escala = 34 / max(ancho, alto)
    nuevo_ancho = max(1, int(ancho * escala))
    nuevo_alto = max(1, int(alto * escala))
    digito = cv2.resize(digito, (nuevo_ancho, nuevo_alto))

    salida = np.zeros((TAMANO_IMAGEN, TAMANO_IMAGEN), dtype=np.uint8)
    x_inicio = (TAMANO_IMAGEN - nuevo_ancho) // 2
    y_inicio = (TAMANO_IMAGEN - nuevo_alto) // 2
    salida[y_inicio : y_inicio + nuevo_alto, x_inicio : x_inicio + nuevo_ancho] = digito

    return salida


def preparar_binaria(gris, tipo_umbral):
    _, binaria = cv2.threshold(
        gris,
        0,
        255,
        tipo_umbral + cv2.THRESH_OTSU,
    )

    binaria = quitar_bordes(binaria)
    binaria = quitar_lineas_de_borde(binaria)
    return binaria


def elegir_mejor_binaria(gris):
    candidatas = [
        preparar_binaria(gris, cv2.THRESH_BINARY_INV),
        preparar_binaria(gris, cv2.THRESH_BINARY),
    ]

    candidatas_validas = []

    for binaria in candidatas:
        proporcion_tinta = np.mean(binaria > 0)

        if 0.01 <= proporcion_tinta <= 0.40:
            candidatas_validas.append((proporcion_tinta, binaria))

    if not candidatas_validas:
        return None

    return min(candidatas_validas, key=lambda candidata: candidata[0])[1]


def limpiar_casilla(casilla):
    gris = convertir_a_gris(casilla)
    gris = cv2.resize(gris, (TAMANO_IMAGEN, TAMANO_IMAGEN))
    gris = cv2.GaussianBlur(gris, (3, 3), 0)

    binaria = elegir_mejor_binaria(gris)

    if binaria is None:
        return np.zeros((TAMANO_IMAGEN, TAMANO_IMAGEN), dtype=np.uint8)

    return centrar_digito(binaria)
