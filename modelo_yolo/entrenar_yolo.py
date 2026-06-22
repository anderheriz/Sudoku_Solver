from multiprocessing import freeze_support
from pathlib import Path

from ultralytics import YOLO


RUTA_MODELO_YOLO = Path(__file__).resolve().parent
RUTA_MODELO_BASE = RUTA_MODELO_YOLO / "yolov8n.pt"
RUTA_DATA = RUTA_MODELO_YOLO / "data.yaml"
RUTA_RUNS = RUTA_MODELO_YOLO / "runs"


def main():
    modelo = YOLO(str(RUTA_MODELO_BASE))

    modelo.train(
        data=str(RUTA_DATA),
        epochs=30,
        batch=32,
        imgsz=640,
        device=0,
        project=str(RUTA_RUNS),
        name="sudoku_gpu",
        save=True,
        exist_ok=True,
        workers=0,
    )


if __name__ == "__main__":
    freeze_support()
    main()
