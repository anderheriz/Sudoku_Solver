VACIO = 0


def obtener_candidatos(tablero, fila, columna):
    if tablero[fila][columna] != VACIO:
        return []

    candidatos = []

    for numero in range(1, 10):
        if es_numero_valido(tablero, fila, columna, numero):
            candidatos.append(numero)

    return candidatos


def es_numero_valido(tablero, fila, columna, numero):
    if numero in tablero[fila]:
        return False

    for i in range(9):
        if tablero[i][columna] == numero:
            return False

    inicio_fila = (fila // 3) * 3
    inicio_columna = (columna // 3) * 3

    for i in range(inicio_fila, inicio_fila + 3):
        for j in range(inicio_columna, inicio_columna + 3):
            if tablero[i][j] == numero:
                return False

    return True


def buscar_mejor_casilla_vacia(tablero):
    mejor_casilla = None
    mejores_candidatos = None

    for fila in range(9):
        for columna in range(9):
            if tablero[fila][columna] == VACIO:
                candidatos = obtener_candidatos(tablero, fila, columna)

                if len(candidatos) == 0:
                    return (fila, columna), []

                if mejores_candidatos is None or len(candidatos) < len(mejores_candidatos):
                    mejor_casilla = (fila, columna)
                    mejores_candidatos = candidatos

    return mejor_casilla, mejores_candidatos


def tablero_inicial_es_valido(tablero):
    for fila in range(9):
        for columna in range(9):
            numero = tablero[fila][columna]

            if numero == VACIO:
                continue

            tablero[fila][columna] = VACIO
            valido = es_numero_valido(tablero, fila, columna, numero)
            tablero[fila][columna] = numero

            if not valido:
                return False

    return True


def resolver_sudoku(tablero, limite_intentos=100000):
    if not tablero_inicial_es_valido(tablero):
        return False

    intentos = [0]
    return resolver_sudoku_recursivo(tablero, intentos, limite_intentos)


def resolver_sudoku_recursivo(tablero, intentos, limite_intentos):
    if intentos[0] > limite_intentos:
        return False

    casilla, candidatos = buscar_mejor_casilla_vacia(tablero)

    if casilla is None:
        return True

    if len(candidatos) == 0:
        return False

    fila, columna = casilla

    for numero in candidatos:
        intentos[0] += 1
        tablero[fila][columna] = numero

        if resolver_sudoku_recursivo(tablero, intentos, limite_intentos):
            return True

        tablero[fila][columna] = VACIO

    return False


def tablero_resuelto_es_valido(tablero):
    for fila in tablero:
        if VACIO in fila:
            return False

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


def imprimir_tablero(tablero):
    for fila in tablero:
        print(fila)


def main():
    tablero = [
        [5, 3, 0, 0, 7, 0, 0, 0, 0],
        [6, 0, 0, 1, 9, 5, 0, 0, 0],
        [0, 9, 8, 0, 0, 0, 0, 6, 0],
        [8, 0, 0, 0, 6, 0, 0, 0, 3],
        [4, 0, 0, 8, 0, 3, 0, 0, 1],
        [7, 0, 0, 0, 2, 0, 0, 0, 6],
        [0, 6, 0, 0, 0, 0, 2, 8, 0],
        [0, 0, 0, 4, 1, 9, 0, 0, 5],
        [0, 0, 0, 0, 8, 0, 0, 7, 9],
    ]

    if resolver_sudoku(tablero) and tablero_resuelto_es_valido(tablero):
        print("Sudoku resuelto:")
        imprimir_tablero(tablero)
    else:
        print("No se ha podido resolver el sudoku.")


if __name__ == "__main__":
    main()
