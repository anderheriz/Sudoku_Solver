# Proyecto Sudoku

Aplicacion de Streamlit que recibe una imagen de un sudoku 9x9, detecta el tablero, lee las casillas y muestra el sudoku resuelto.

La version final de la app funciona de forma secuencial: al subir una imagen se detecta el tablero, se lee con la CNN y se intenta resolver automaticamente. En la solucion final, los numeros originales se muestran en color oscuro y los numeros anadidos por el modelo/reglas se muestran en verde.

## Estructura del proyecto

- `app/`: aplicacion final de Streamlit.
- `modelo_yolo/`: entrenamiento y modelo YOLO para detectar el tablero.
- `modelo_casillas/`: recorte del tablero, preprocesado de casillas y CNN de clasificacion de numeros.
- `modelo_juego/`: modelo neuronal de solucion y validacion con reglas de sudoku.

## Pipeline

1. El usuario sube una imagen en Streamlit.
2. YOLO detecta la zona del tablero.
3. OpenCV ajusta el recorte y divide el tablero en 81 casillas.
4. La CNN clasifica cada casilla como vacia (`0`) o como un numero del `1` al `9`.
5. Se valida que la lectura no contradiga las reglas de Sudoku.
6. El modelo neuronal de solucion propone probabilidades y una capa de reglas completa el tablero.
7. Streamlit muestra el sudoku resuelto, marcando en verde los numeros anadidos.

## Ejecutar la app

Desde la carpeta raiz del proyecto:

```powershell
pip install -r requirements.txt
streamlit run app/main.py
```

## Modelos necesarios para la demo

Para ejecutar la app final deben existir estos archivos:

- `modelo_yolo/modelos/best.pt`
- `modelo_casillas/modelos/cnn_casillas.keras`
- `modelo_juego/modelos/modelo_solucion.keras`

Tambien se conservan las graficas y metricas de entrenamiento en las carpetas `modelos/`.

## Resultados actuales

### CNN de casillas

Modelo encargado de clasificar cada recorte de casilla como vacia (`0`) o como un numero del `1` al `9`.

- Mejor `val_accuracy`: `0.6279`
- Ultima `val_accuracy`: `0.6270`
- Epocas: `10`
- Batch size: `64`

El modelo se ha entrenado combinando ejemplos reales recortados de sudokus, aumentos de datos y datasets de digitos.

### Modelo de solucion

Modelo entrenado con `1.000.000` de sudokus del archivo `sudoku.csv`.

- `Test accuracy`: `0.4588`
- Accuracy por casilla: `0.4592`
- Accuracy en huecos originales: `0.2427`
- Accuracy en pistas originales: `0.6930`
- Accuracy de tablero completo: `0.0000`

Este modelo no se usa como solucionador final puro. Se utiliza como apoyo para proponer probabilidades, y despues una capa de reglas de Sudoku valida filas, columnas y bloques 3x3.

## Entrenamiento

### YOLO

```powershell
cd modelo_yolo
python entrenar_yolo.py
```

El entrenamiento genera resultados en `modelo_yolo/runs/`. Esa carpeta esta ignorada en Git porque es una salida generada.

### CNN de casillas

```powershell
cd modelo_casillas
python crear_dataset_cnn_combinado.py
python entrenar_cnn.py
```

El dataset combinado se genera en `modelo_casillas/dataset_cnn_combinado/`, carpeta ignorada en Git.

### Modelo de solucion

```powershell
cd modelo_juego
python entrenar_modelo_solucion.py
```

Para entrenamientos pesados se recomienda usar Google Colab o una GPU local.

## Archivos grandes

Los datasets grandes no se deben subir a GitHub:

- `modelo_juego/dataset/sudoku.csv`
- `modelo_juego/dataset/sudoku_*.csv`
- `modelo_yolo/imag_proc/`
- `modelo_casillas/dataset_cnn_combinado/`

Estos archivos son necesarios para reentrenar, pero no para ejecutar la demo final si ya estan guardados los tres modelos entrenados.

## Notas

- Si la CNN lee mal una casilla, la app muestra avisos de contradiccion cuando detecta numeros repetidos en filas, columnas o bloques.
- El preprocesado de casillas intenta normalizar imagenes claras, oscuras y con distintos tipos de fuente.
- Para mejorar el proyecto, el siguiente paso natural seria guardar casillas de baja confianza y usarlas para reentrenar la CNN.
