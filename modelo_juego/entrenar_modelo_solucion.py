from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras import layers, models

from dataset_sudoku import preparar_datos


RUTA_PROYECTO = Path(__file__).resolve().parents[1]
CARPETA_MODELOS = RUTA_PROYECTO / "modelo_juego/modelos"
RUTA_MODELO = CARPETA_MODELOS / "modelo_solucion.keras"
RUTA_GRAFICA = CARPETA_MODELOS / "accuracy_modelo_solucion.png"
RUTA_METRICAS = CARPETA_MODELOS / "metricas_modelo_solucion.txt"

N_FILAS = None
TEST_SIZE = 0.2
RANDOM_STATE = 42
BATCH_SIZE = 256
EPOCHS = 30


def crear_modelo():
    modelo = models.Sequential(
        [
            layers.Input(shape=(81,), dtype="int32"),
            layers.Embedding(input_dim=10, output_dim=32),
            layers.LSTM(256, return_sequences=True),
            layers.LSTM(256, return_sequences=True),
            layers.TimeDistributed(layers.Dense(9, activation="softmax")),
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
    plt.title("Accuracy modelo solucion")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.tight_layout()
    plt.savefig(RUTA_GRAFICA)
    plt.close()


def calcular_metricas_exactas(modelo, x_test, y_test, max_ejemplos=5000):
    x_muestra = x_test[:max_ejemplos]
    y_muestra = y_test[:max_ejemplos]

    predicciones = modelo.predict(x_muestra, verbose=False)
    y_pred = np.argmax(predicciones, axis=-1)

    accuracy_casillas = float(np.mean(y_pred == y_muestra))
    mascara_huecos = x_muestra == 0
    mascara_pistas = x_muestra != 0

    accuracy_huecos = float(np.mean(y_pred[mascara_huecos] == y_muestra[mascara_huecos]))
    accuracy_pistas = float(np.mean(y_pred[mascara_pistas] == y_muestra[mascara_pistas]))
    accuracy_tableros = float(np.mean(np.all(y_pred == y_muestra, axis=1)))

    return accuracy_casillas, accuracy_huecos, accuracy_pistas, accuracy_tableros


def guardar_metricas(
    test_loss,
    test_accuracy,
    accuracy_casillas,
    accuracy_huecos,
    accuracy_pistas,
    accuracy_tableros,
):
    texto = [
        f"Filas usadas del CSV: {N_FILAS}",
        f"Test size: {TEST_SIZE}",
        f"Test loss: {test_loss:.4f}",
        f"Test accuracy por casilla: {test_accuracy:.4f}",
        f"Accuracy exacta por casilla: {accuracy_casillas:.4f}",
        f"Accuracy en huecos originales: {accuracy_huecos:.4f}",
        f"Accuracy en pistas originales: {accuracy_pistas:.4f}",
        f"Accuracy de tablero completo: {accuracy_tableros:.4f}",
    ]

    RUTA_METRICAS.write_text("\n".join(texto), encoding="utf-8")


def main():
    CARPETA_MODELOS.mkdir(parents=True, exist_ok=True)

    print("Cargando sudoku.csv...")
    x, y = preparar_datos(n_filas=N_FILAS)

    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
    )

    print(f"Train: {len(x_train)} ejemplos")
    print(f"Test: {len(x_test)} ejemplos")

    modelo = crear_modelo()

    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=4,
            restore_best_weights=True,
        )
    ]

    historial = modelo.fit(
        x_train,
        y_train,
        validation_split=0.2,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=callbacks,
    )

    test_loss, test_accuracy = modelo.evaluate(x_test, y_test, verbose=False)
    (
        accuracy_casillas,
        accuracy_huecos,
        accuracy_pistas,
        accuracy_tableros,
    ) = calcular_metricas_exactas(modelo, x_test, y_test)

    modelo.save(RUTA_MODELO)
    guardar_grafica(historial)
    guardar_metricas(
        test_loss,
        test_accuracy,
        accuracy_casillas,
        accuracy_huecos,
        accuracy_pistas,
        accuracy_tableros,
    )

    print("Modelo de solucion guardado correctamente.")
    print(f"Ruta: {RUTA_MODELO}")
    print(f"Accuracy test por casilla: {test_accuracy:.4f}")
    print(f"Accuracy en huecos originales: {accuracy_huecos:.4f}")
    print(f"Accuracy tablero completo: {accuracy_tableros:.4f}")


if __name__ == "__main__":
    main()
