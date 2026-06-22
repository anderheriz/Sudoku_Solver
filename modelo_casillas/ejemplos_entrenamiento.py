from pathlib import Path
import sys

import cv2

from recortar_casillas import cargar_modelo_yolo, dividir_en_casillas, recortar_tablero


RUTA_PROYECTO = Path(__file__).resolve().parents[1]
CARPETA_ENTRENAMIENTO = RUTA_PROYECTO / "modelo_casillas/casillas_entrenamiento"


def guardar_casillas_para_entrenar(ruta_imagen, nombre_ejemplo):
    carpeta_salida = CARPETA_ENTRENAMIENTO / nombre_ejemplo
    carpeta_salida.mkdir(parents=True, exist_ok=True)

    modelo_yolo = cargar_modelo_yolo()
    tablero = recortar_tablero(modelo_yolo, ruta_imagen)
    casillas = dividir_en_casillas(tablero)

    cv2.imwrite(str(carpeta_salida / "tablero_recortado.jpg"), tablero)

    for indice, casilla in enumerate(casillas):
        fila = indice // 9
        columna = indice % 9
        ruta_casilla = carpeta_salida / f"casilla_{fila:02d}_{columna:02d}.jpg"
        cv2.imwrite(str(ruta_casilla), casilla)

    print("Ejemplo preparado correctamente.")
    print(f"Carpeta: {carpeta_salida}")
    print("Ahora etiqueta este tablero en crear_dataset_real_aumentado.py.")
    print("Plantilla:")
    print("[")
    for _ in range(9):
        print("    [0, 0, 0, 0, 0, 0, 0, 0, 0],")
    print("]")


def main():
    if len(sys.argv) < 3:
        print("Uso: python preparar_ejemplo_entrenamiento.py ruta/imagen.jpg nombre_ejemplo")
        return

    ruta_imagen = Path(sys.argv[1])
    nombre_ejemplo = sys.argv[2]
    guardar_casillas_para_entrenar(ruta_imagen, nombre_ejemplo)


if __name__ == "__main__":
    main()
