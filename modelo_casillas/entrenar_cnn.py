from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import cv2
import numpy as np
from sklearn.datasets import load_digits
import tensorflow as tf
from tensorflow.keras import layers, models

from preprocesar_casilla import limpiar_casilla


RUTA_PROYECTO = Path(__file__).resolve().parents[1]

NOMBRE_DATASET = "dataset_cnn_combinado"
CARPETA_DATASET = RUTA_PROYECTO / f"modelo_casillas/{NOMBRE_DATASET}"
CARPETA_MODELOS = RUTA_PROYECTO / "modelo_casillas/modelos"
RUTA_MODELO_SALIDA = CARPETA_MODELOS / "cnn_casillas.keras"

TAMANO_IMAGEN = (50, 50)
BATCH_SIZE = 64
EPOCHS = 10
MAX_MNIST_POR_DIGITO = 3000


def cargar_imagenes_split(split):
    imagenes = []
    etiquetas = []

    for etiqueta in range(10):
        carpeta = CARPETA_DATASET / split / str(etiqueta)

        for ruta_imagen in carpeta.glob("*.jpg"):
            imagen = cv2.imread(str(ruta_imagen), cv2.IMREAD_GRAYSCALE)

            if imagen is None:
                continue

            imagen = limpiar_casilla(imagen)
            imagenes.append(imagen)
            etiquetas.append(etiqueta)

    x = np.array(imagenes, dtype=np.uint8).reshape(-1, 50, 50, 1)
    y = np.array(etiquetas, dtype=np.int64)

    return x, y


def aumentar_digito(imagen, indice):
    alto, ancho = imagen.shape
    centro = (ancho // 2, alto // 2)
    angulo = (indice % 7) - 3
    escala = 0.9 + (indice % 5) * 0.04
    matriz = cv2.getRotationMatrix2D(centro, angulo, escala)
    matriz[0, 2] += (indice % 3) - 1
    matriz[1, 2] += ((indice // 3) % 3) - 1
    return cv2.warpAffine(imagen, matriz, (ancho, alto), borderValue=0)


def cargar_digitos_sklearn():
    datos = load_digits()
    imagenes = []
    etiquetas = []

    for imagen, etiqueta in zip(datos.images, datos.target):
        if etiqueta == 0:
            continue

        imagen = (imagen / imagen.max() * 255).astype(np.uint8)
        imagen = cv2.resize(imagen, TAMANO_IMAGEN)

        for indice in range(8):
            imagenes.append(aumentar_digito(imagen, indice))
            etiquetas.append(int(etiqueta))

    x = np.array(imagenes, dtype=np.uint8).reshape(-1, 50, 50, 1)
    y = np.array(etiquetas, dtype=np.int64)

    return x, y


def convertir_mnist_a_50x50(imagen):
    imagen = imagen.astype(np.uint8)
    imagen = cv2.resize(imagen, TAMANO_IMAGEN)
    return limpiar_casilla(imagen)


def cargar_digitos_mnist():
    (x_train, y_train), _ = tf.keras.datasets.mnist.load_data()
    imagenes = []
    etiquetas = []
    contador_por_digito = {digito: 0 for digito in range(1, 10)}

    for imagen, etiqueta in zip(x_train, y_train):
        etiqueta = int(etiqueta)

        if etiqueta == 0:
            continue

        if contador_por_digito[etiqueta] >= MAX_MNIST_POR_DIGITO:
            continue

        imagen = convertir_mnist_a_50x50(imagen)
        imagenes.append(imagen)
        etiquetas.append(etiqueta)
        contador_por_digito[etiqueta] += 1

        if all(cantidad >= MAX_MNIST_POR_DIGITO for cantidad in contador_por_digito.values()):
            break

    x = np.array(imagenes, dtype=np.uint8).reshape(-1, 50, 50, 1)
    y = np.array(etiquetas, dtype=np.int64)

    return x, y


def cargar_datasets():
    x_train, y_train = cargar_imagenes_split("train")
    x_val, y_val = cargar_imagenes_split("val")

    try:
        x_digits, y_digits = cargar_digitos_mnist()
        print("Dataset extra: MNIST")
    except Exception as error:
        print(f"No se ha podido cargar MNIST. Uso sklearn digits. Motivo: {error}")
        x_digits, y_digits = cargar_digitos_sklearn()
        print("Dataset extra: sklearn digits")

    x_train = np.concatenate([x_train, x_digits], axis=0)
    y_train = np.concatenate([y_train, y_digits], axis=0)

    print(f"Train: {len(x_train)} imagenes")
    print(f"Val: {len(x_val)} imagenes")

    return x_train, y_train, x_val, y_val


def calcular_pesos_clase(y_train):
    clases, cuentas = np.unique(y_train, return_counts=True)
    total = len(y_train)
    pesos = {}

    for clase, cuenta in zip(clases, cuentas):
        pesos[int(clase)] = total / (len(clases) * int(cuenta))

    return pesos


def crear_modelo():
    modelo = models.Sequential(
        [
            layers.Input(shape=(50, 50, 1)),
            layers.Rescaling(1.0 / 255),
            layers.Conv2D(32, (3, 3), activation="relu"),
            layers.MaxPooling2D((2, 2)),
            layers.Conv2D(64, (3, 3), activation="relu"),
            layers.MaxPooling2D((2, 2)),
            layers.Flatten(),
            layers.Dense(64, activation="relu"),
            layers.Dropout(0.2),
            layers.Dense(10, activation="softmax"),
        ]
    )

    modelo.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    return modelo


def guardar_grafica(historial):
    plt.figure(figsize=(8, 4))
    plt.plot(historial.history["accuracy"], label="train")
    plt.plot(historial.history["val_accuracy"], label="val")
    plt.title("Accuracy CNN casillas")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.tight_layout()
    plt.savefig(CARPETA_MODELOS / "accuracy_cnn.png")
    plt.close()


def guardar_metricas(historial, pesos_clase):
    ruta_metricas = CARPETA_MODELOS / "metricas_cnn.txt"
    mejor_val_accuracy = max(historial.history["val_accuracy"])
    ultima_val_accuracy = historial.history["val_accuracy"][-1]

    with open(ruta_metricas, "w", encoding="utf-8") as archivo:
        archivo.write(f"Mejor val_accuracy: {mejor_val_accuracy:.4f}\n")
        archivo.write(f"Ultima val_accuracy: {ultima_val_accuracy:.4f}\n")
        archivo.write(f"Batch size: {BATCH_SIZE}\n")
        archivo.write(f"Epochs: {EPOCHS}\n")
        archivo.write(f"Pesos de clase: {pesos_clase}\n")


def main():
    CARPETA_MODELOS.mkdir(parents=True, exist_ok=True)

    x_train, y_train, x_val, y_val = cargar_datasets()
    modelo = crear_modelo()
    pesos_clase = calcular_pesos_clase(y_train)

    historial = modelo.fit(
        x_train,
        y_train,
        validation_data=(x_val, y_val),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        shuffle=True,
        class_weight=pesos_clase,
    )

    modelo.save(RUTA_MODELO_SALIDA)
    guardar_grafica(historial)
    guardar_metricas(historial, pesos_clase)

    print("Modelo CNN guardado correctamente.")
    print(f"Ruta: {RUTA_MODELO_SALIDA}")


if __name__ == "__main__":
    main()
