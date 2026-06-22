from pathlib import Path
import random
import shutil

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from crear_dataset_real_aumentado import EJEMPLOS_ENTRENAMIENTO, aumentar_imagen


RUTA_PROYECTO = Path(__file__).resolve().parents[1]
CARPETA_DATASET = RUTA_PROYECTO / "modelo_casillas/dataset_cnn_combinado"
CARPETA_FUENTES = Path("C:/Windows/Fonts")

TAMANO_IMAGEN = 50
IMAGENES_FUENTES_TRAIN = 500
IMAGENES_FUENTES_VAL = 120
AUMENTOS_REALES_POR_CASILLA = 25


FUENTES_PREFERIDAS = [
    "arial.ttf",
    "arialbd.ttf",
    "calibri.ttf",
    "calibrib.ttf",
    "cambria.ttc",
    "cambriab.ttf",
    "consola.ttf",
    "consolab.ttf",
    "cour.ttf",
    "courbd.ttf",
    "georgia.ttf",
    "georgiab.ttf",
    "segoeui.ttf",
    "segoeuib.ttf",
    "tahoma.ttf",
    "tahomabd.ttf",
    "times.ttf",
    "timesbd.ttf",
    "verdana.ttf",
    "verdanab.ttf",
]


def buscar_fuentes():
    fuentes = []

    for nombre in FUENTES_PREFERIDAS:
        ruta = CARPETA_FUENTES / nombre

        if ruta.exists():
            fuentes.append(ruta)

    if not fuentes:
        fuentes = list(CARPETA_FUENTES.glob("*.ttf"))[:20]

    return fuentes


def carpeta_clase(split, etiqueta):
    carpeta = CARPETA_DATASET / split / str(etiqueta)
    carpeta.mkdir(parents=True, exist_ok=True)
    return carpeta


def guardar_imagen(imagen, split, etiqueta, nombre):
    ruta = carpeta_clase(split, etiqueta) / f"{nombre}.jpg"
    cv2.imwrite(str(ruta), imagen)


def dibujar_lineas_casilla(draw, rng):
    color = rng.randint(20, 110)

    for lado in ["arriba", "abajo", "izquierda", "derecha"]:
        if rng.random() > 0.35:
            continue

        grosor = rng.randint(1, 4)

        if lado == "arriba":
            draw.rectangle([0, 0, TAMANO_IMAGEN, grosor], fill=color)
        elif lado == "abajo":
            draw.rectangle([0, TAMANO_IMAGEN - grosor, TAMANO_IMAGEN, TAMANO_IMAGEN], fill=color)
        elif lado == "izquierda":
            draw.rectangle([0, 0, grosor, TAMANO_IMAGEN], fill=color)
        else:
            draw.rectangle([TAMANO_IMAGEN - grosor, 0, TAMANO_IMAGEN, TAMANO_IMAGEN], fill=color)


def aplicar_ruido(imagen, rng):
    imagen_np = np.array(imagen).astype(np.int16)

    brillo = rng.randint(-18, 18)
    imagen_np = np.clip(imagen_np + brillo, 0, 255)

    ruido = rng.normalvariate(0, 4)
    if abs(ruido) > 1:
        matriz_ruido = np.random.normal(0, abs(ruido), imagen_np.shape)
        imagen_np = np.clip(imagen_np + matriz_ruido, 0, 255)

    imagen_np = imagen_np.astype(np.uint8)

    if rng.random() < 0.25:
        imagen_np = cv2.GaussianBlur(imagen_np, (3, 3), 0)

    return imagen_np


def crear_casilla_fuente(etiqueta, fuentes, rng):
    fondo = rng.randint(235, 255)
    imagen = Image.new("L", (TAMANO_IMAGEN, TAMANO_IMAGEN), color=fondo)
    draw = ImageDraw.Draw(imagen)

    dibujar_lineas_casilla(draw, rng)

    if etiqueta != 0:
        fuente = rng.choice(fuentes)
        tamano_fuente = rng.randint(24, 36)
        font = ImageFont.truetype(str(fuente), tamano_fuente)
        texto = str(etiqueta)
        caja = draw.textbbox((0, 0), texto, font=font)
        ancho_texto = caja[2] - caja[0]
        alto_texto = caja[3] - caja[1]

        x = (TAMANO_IMAGEN - ancho_texto) // 2 + rng.randint(-4, 4)
        y = (TAMANO_IMAGEN - alto_texto) // 2 + rng.randint(-5, 5)
        color_texto = rng.randint(10, 80)
        draw.text((x, y), texto, fill=color_texto, font=font)

    return aplicar_ruido(imagen, rng)


def generar_dataset_fuentes():
    rng = random.Random(123)
    fuentes = buscar_fuentes()
    contador = 0

    for etiqueta in range(10):
        for indice in range(IMAGENES_FUENTES_TRAIN):
            imagen = crear_casilla_fuente(etiqueta, fuentes, rng)
            guardar_imagen(imagen, "train", etiqueta, f"fuente_{etiqueta}_{indice:05d}")
            contador += 1

        for indice in range(IMAGENES_FUENTES_VAL):
            imagen = crear_casilla_fuente(etiqueta, fuentes, rng)
            guardar_imagen(imagen, "val", etiqueta, f"fuente_{etiqueta}_{indice:05d}")
            contador += 1

    return contador, len(fuentes)


def generar_dataset_real():
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

                for _ in range(AUMENTOS_REALES_POR_CASILLA):
                    split = "train" if contador % 5 != 0 else "val"
                    imagen_aumentada = aumentar_imagen(imagen)
                    nombre = f"real_{etiqueta}_{contador:05d}"
                    guardar_imagen(imagen_aumentada, split, etiqueta, nombre)
                    contador += 1

    return contador


def main():
    if CARPETA_DATASET.exists():
        shutil.rmtree(CARPETA_DATASET)

    imagenes_fuentes, numero_fuentes = generar_dataset_fuentes()
    imagenes_reales = generar_dataset_real()

    print("Dataset CNN combinado creado correctamente.")
    print(f"Fuentes usadas: {numero_fuentes}")
    print(f"Imagenes generadas con fuentes: {imagenes_fuentes}")
    print(f"Imagenes reales aumentadas: {imagenes_reales}")
    print(f"Total imagenes: {imagenes_fuentes + imagenes_reales}")
    print(f"Carpeta: {CARPETA_DATASET}")


if __name__ == "__main__":
    main()
