from dataset_sudoku import cargar_ejemplo, tablero_a_texto
from predecir_solucion import predecir_solucion, tablero_resuelto_es_valido


def main():
    puzzle, solucion_real = cargar_ejemplo(indice=0)
    solucion_predicha = predecir_solucion(puzzle)

    print("Puzzle inicial:")
    for fila in puzzle:
        print(fila)

    print("\nSolucion del dataset:")
    for fila in solucion_real:
        print(fila)

    print("\nSolucion predicha por el modelo:")
    if solucion_predicha is None:
        print("El modelo no ha generado una solucion valida.")
        coincide = False
        valido = False
    else:
        for fila in solucion_predicha:
            print(fila)

        coincide = tablero_a_texto(solucion_predicha) == tablero_a_texto(solucion_real)
        valido = tablero_resuelto_es_valido(solucion_predicha)

    print(f"Valido: {valido}")
    print(f"Coincide con el CSV: {coincide}")


if __name__ == "__main__":
    main()
