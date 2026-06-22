from pathlib import Path

import numpy as np
import tensorflow as tf


RUTA_PROYECTO = Path(__file__).resolve().parents[1]
RUTA_MODELO = RUTA_PROYECTO / "modelo_juego/modelos/modelo_solucion.keras"
MINIMO_PISTAS = 17
LIMITE_INTENTOS = 100000


def cargar_modelo_solucion():
    if not RUTA_MODELO.exists():
        raise FileNotFoundError(
            "No existe el modelo de solucion. Ejecuta primero "
            "modelo_juego/entrenar_modelo_solucion.py"
        )

    return tf.keras.models.load_model(RUTA_MODELO)


def tablero_a_entrada(tablero, modelo=None):
    valores = [numero for fila in tablero for numero in fila]
    entrada = np.array(valores, dtype=np.int64).reshape(1, 81)

    if modelo is not None and "int" not in str(modelo.inputs[0].dtype):
        return entrada.astype(np.float32) / 9.0

    return entrada


def contar_pistas(tablero):
    return sum(numero != 0 for fila in tablero for numero in fila)


def valores_sin_repetidos(valores):
    numeros = [valor for valor in valores if valor != 0]
    return len(numeros) == len(set(numeros))


def validar_tablero_inicial(tablero):
    for fila in tablero:
        if not valores_sin_repetidos(fila):
            return False

    for columna in range(9):
        valores_columna = [tablero[fila][columna] for fila in range(9)]
        if not valores_sin_repetidos(valores_columna):
            return False

    for inicio_fila in range(0, 9, 3):
        for inicio_columna in range(0, 9, 3):
            valores_caja = []

            for fila in range(inicio_fila, inicio_fila + 3):
                for columna in range(inicio_columna, inicio_columna + 3):
                    valores_caja.append(tablero[fila][columna])

            if not valores_sin_repetidos(valores_caja):
                return False

    return True


def tablero_resuelto_es_valido(tablero):
    numeros = set(range(1, 10))

    for fila in tablero:
        if set(fila) != numeros:
            return False

    for columna in range(9):
        valores_columna = {tablero[fila][columna] for fila in range(9)}
        if valores_columna != numeros:
            return False

    for inicio_fila in range(0, 9, 3):
        for inicio_columna in range(0, 9, 3):
            valores_caja = set()

            for fila in range(inicio_fila, inicio_fila + 3):
                for columna in range(inicio_columna, inicio_columna + 3):
                    valores_caja.add(tablero[fila][columna])

            if valores_caja != numeros:
                return False

    return True


def es_numero_valido(tablero, fila, columna, numero):
    if numero in tablero[fila]:
        return False

    for indice_fila in range(9):
        if tablero[indice_fila][columna] == numero:
            return False

    inicio_fila = (fila // 3) * 3
    inicio_columna = (columna // 3) * 3

    for indice_fila in range(inicio_fila, inicio_fila + 3):
        for indice_columna in range(inicio_columna, inicio_columna + 3):
            if tablero[indice_fila][indice_columna] == numero:
                return False

    return True


def obtener_candidatos(tablero, fila, columna):
    candidatos = []

    for numero in range(1, 10):
        if es_numero_valido(tablero, fila, columna, numero):
            candidatos.append(numero)

    return candidatos


def obtener_probabilidades_modelo(tablero, modelo):
    entrada = tablero_a_entrada(tablero, modelo)
    prediccion = modelo.predict(entrada, verbose=False)[0]
    return prediccion


def obtener_probabilidad_numero(probabilidades, indice, numero):
    if probabilidades.shape[-1] == 10:
        return probabilidades[indice][numero]

    return probabilidades[indice][numero - 1]


def ordenar_candidatos_por_modelo(candidatos, probabilidades, fila, columna):
    indice = fila * 9 + columna

    return sorted(
        candidatos,
        key=lambda numero: obtener_probabilidad_numero(probabilidades, indice, numero),
        reverse=True,
    )


def buscar_mejor_casilla_vacia(tablero, probabilidades):
    mejor_casilla = None
    mejores_candidatos = None

    for fila in range(9):
        for columna in range(9):
            if tablero[fila][columna] != 0:
                continue

            candidatos = obtener_candidatos(tablero, fila, columna)
            candidatos = ordenar_candidatos_por_modelo(candidatos, probabilidades, fila, columna)

            if len(candidatos) == 0:
                return (fila, columna), []

            if mejores_candidatos is None or len(candidatos) < len(mejores_candidatos):
                mejor_casilla = (fila, columna)
                mejores_candidatos = candidatos

    return mejor_casilla, mejores_candidatos


def resolver_con_reglas_y_modelo(tablero, probabilidades, intentos):
    if intentos[0] > LIMITE_INTENTOS:
        return False

    casilla, candidatos = buscar_mejor_casilla_vacia(tablero, probabilidades)

    if casilla is None:
        return True

    if len(candidatos) == 0:
        return False

    fila, columna = casilla

    for numero in candidatos:
        intentos[0] += 1
        tablero[fila][columna] = numero

        if resolver_con_reglas_y_modelo(tablero, probabilidades, intentos):
            return True

        tablero[fila][columna] = 0

    return False


def predecir_solucion(tablero, modelo=None):
    if contar_pistas(tablero) < MINIMO_PISTAS:
        return None

    if not validar_tablero_inicial(tablero):
        return None

    if modelo is None:
        modelo = cargar_modelo_solucion()

    probabilidades = obtener_probabilidades_modelo(tablero, modelo)
    solucion = [fila.copy() for fila in tablero]
    intentos = [0]

    if resolver_con_reglas_y_modelo(solucion, probabilidades, intentos) and tablero_resuelto_es_valido(solucion):
        return solucion

    return None


def imprimir_matriz(matriz):
    for fila in matriz:
        print(fila)


def main():
    tablero = [
        [0, 7, 0, 0, 0, 0, 0, 4, 3],
        [0, 4, 0, 0, 0, 9, 6, 1, 0],
        [8, 0, 0, 6, 3, 4, 9, 0, 0],
        [0, 9, 4, 0, 5, 2, 0, 0, 0],
        [3, 5, 8, 4, 6, 0, 0, 2, 0],
        [0, 0, 0, 8, 0, 0, 5, 3, 0],
        [0, 8, 0, 0, 7, 0, 0, 9, 1],
        [9, 0, 2, 1, 0, 0, 0, 0, 5],
        [0, 0, 7, 0, 4, 0, 8, 0, 2],
    ]

    solucion = predecir_solucion(tablero)

    if solucion is None:
        print("El modelo no ha generado una solucion valida.")
    else:
        imprimir_matriz(solucion)


if __name__ == "__main__":
    main()
