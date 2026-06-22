from pathlib import Path
import hashlib
import sys
import tempfile

import cv2
import pandas as pd
import streamlit as st
from PIL import Image


RUTA_PROYECTO = Path(__file__).resolve().parents[1]
sys.path.append(str(RUTA_PROYECTO / "modelo_casillas"))
sys.path.append(str(RUTA_PROYECTO / "modelo_juego"))

from recortar_casillas import cargar_modelo_yolo, dividir_en_casillas, recortar_tablero
from predecir_casillas import cargar_modelo_cnn, predecir_tablero, resolver_tablero
from predecir_solucion import cargar_modelo_solucion, validar_tablero_inicial


st.set_page_config(page_title="Proyecto Sudoku", layout="centered")

UMBRAL_AVISO_CONFIANZA = 0.65

st.markdown(
    """
    <style>
        h1, h2, h3 {
            text-align: center;
        }
        .separador-seccion {
            border: 0;
            border-top: 2px solid #d9dee7;
            margin: 2rem 0 1.5rem 0;
        }
        div[data-testid="stFileUploader"] label p {
            font-size: 1.15rem;
            font-weight: 500;
        }
        .titulo-tablero-cnn {
            font-size: 2rem;
            font-weight: 700;
            text-align: center;
            margin: 1.25rem 0 1rem 0;
        }
        .sudoku-lectura {
            width: 100%;
            table-layout: fixed;
            border-collapse: collapse;
            margin-bottom: 1rem;
        }
        .sudoku-lectura td {
            border: 1px solid #e5e7eb;
            text-align: center;
            vertical-align: middle;
            padding: 0.55rem 0;
            font-size: 1.05rem;
            color: #30323d;
        }
        .sudoku-lectura td:nth-child(3),
        .sudoku-lectura td:nth-child(6) {
            border-right: 2px solid #b8beca;
        }
        .sudoku-lectura tr:nth-child(3) td,
        .sudoku-lectura tr:nth-child(6) td {
            border-bottom: 2px solid #b8beca;
        }
        .sudoku-solucion {
            width: 100%;
            table-layout: fixed;
            border-collapse: collapse;
            margin-top: 1rem;
        }
        .sudoku-solucion td {
            border: 1px solid #e5e7eb;
            text-align: center;
            vertical-align: middle;
            padding: 0.75rem 0;
            font-size: 1.3rem;
            color: #30323d;
        }
        .sudoku-solucion td:nth-child(3),
        .sudoku-solucion td:nth-child(6) {
            border-right: 2px solid #b8beca;
        }
        .sudoku-solucion tr:nth-child(3) td,
        .sudoku-solucion tr:nth-child(6) td {
            border-bottom: 2px solid #b8beca;
        }
        .sudoku-solucion .numero-solucion {
            color: #15803d;
            font-weight: 700;
        }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource
def cargar_modelos():
    modelo_yolo = cargar_modelo_yolo()
    modelo_cnn = cargar_modelo_cnn()
    modelo_solucion = cargar_modelo_solucion()
    return modelo_yolo, modelo_cnn, modelo_solucion


def guardar_imagen_temporal(imagen_subida):
    extension = Path(imagen_subida.name).suffix

    with tempfile.NamedTemporaryFile(delete=False, suffix=extension) as archivo:
        archivo.write(imagen_subida.getbuffer())
        return Path(archivo.name)


def obtener_clave_imagen(imagen_subida):
    return hashlib.md5(imagen_subida.getvalue()).hexdigest()


def limpiar_resultados_anteriores():
    claves = [
        "tablero_recortado",
        "tablero_predicho",
        "confianzas",
        "tablero_resuelto",
        "error_solucion",
        "error_proceso",
    ]

    for clave in claves:
        st.session_state.pop(clave, None)


def titulo_seccion(texto):
    st.markdown(f"### {texto}")


def separador():
    st.markdown("<hr class='separador-seccion'>", unsafe_allow_html=True)


def mostrar_tablero_leido_cnn(tablero):
    st.markdown('<p class="titulo-tablero-cnn">Tablero leido por CNN</p>', unsafe_allow_html=True)

    filas_html = []
    for fila in tablero:
        celdas_html = []
        for numero in fila:
            valor = "" if numero == 0 else str(numero)
            celdas_html.append(f"<td>{valor}</td>")

        filas_html.append("<tr>" + "".join(celdas_html) + "</tr>")

    tabla_html = '<table class="sudoku-lectura"><tbody>'
    tabla_html += "".join(filas_html)
    tabla_html += "</tbody></table>"

    st.markdown(tabla_html, unsafe_allow_html=True)


def mostrar_sudoku_resuelto(tablero_inicial, tablero_resuelto):
    titulo_seccion("Sudoku resuelto")

    filas_html = []
    for fila in range(9):
        celdas_html = []
        for columna in range(9):
            numero = tablero_resuelto[fila][columna]
            clase = ""
            if tablero_inicial[fila][columna] == 0:
                clase = "numero-solucion"

            celdas_html.append(f'<td class="{clase}">{numero}</td>')

        filas_html.append("<tr>" + "".join(celdas_html) + "</tr>")

    tabla_html = '<table class="sudoku-solucion"><tbody>'
    tabla_html += "".join(filas_html)
    tabla_html += "</tbody></table>"

    st.markdown(tabla_html, unsafe_allow_html=True)


def contar_pistas(tablero):
    return sum(numero != 0 for fila in tablero for numero in fila)


def buscar_casillas_baja_confianza(tablero, confianzas):
    casillas = []

    for fila in range(9):
        for columna in range(9):
            numero = tablero[fila][columna]
            confianza = confianzas[fila][columna]

            if numero != 0 and confianza < UMBRAL_AVISO_CONFIANZA:
                casillas.append(
                    {
                        "fila": fila + 1,
                        "columna": columna + 1,
                        "numero": numero,
                        "confianza": confianza,
                    }
                )

    return casillas


def mostrar_diagnostico_lectura(tablero, confianzas):
    pistas = contar_pistas(tablero)

    if pistas < 17:
        st.warning("La CNN ha leido menos de 17 numeros. Prueba con una imagen mas clara o cercana.")

    if not validar_tablero_inicial(tablero):
        st.error("La lectura inicial tiene contradicciones: hay numeros repetidos en alguna fila, columna o bloque.")

    casillas_baja_confianza = buscar_casillas_baja_confianza(tablero, confianzas)

    if casillas_baja_confianza:
        st.warning("Hay casillas con baja confianza. La solucion puede fallar si algun numero se ha leido mal.")
        st.dataframe(pd.DataFrame(casillas_baja_confianza), width="stretch")


def leer_imagen(ruta_imagen):
    modelo_yolo, modelo_cnn, _ = cargar_modelos()

    tablero_recortado = recortar_tablero(modelo_yolo, ruta_imagen)
    casillas = dividir_en_casillas(tablero_recortado)
    tablero_predicho, confianzas = predecir_tablero(casillas=casillas, modelo=modelo_cnn)

    return tablero_recortado, tablero_predicho, confianzas


def resolver_tablero_leido(tablero):
    pistas = contar_pistas(tablero)

    if pistas < 17:
        return None, "El tablero tiene muy pocos numeros leidos. Prueba con una imagen mas clara o cercana."

    if not validar_tablero_inicial(tablero):
        return None, "El tablero tiene numeros repetidos en alguna fila, columna o bloque 3x3."

    _, _, modelo_solucion = cargar_modelos()
    tablero_resuelto = resolver_tablero(tablero, modelo_solucion=modelo_solucion)

    if tablero_resuelto is None:
        return None, "No se ha podido resolver el sudoku con los numeros actuales."

    return tablero_resuelto, None


st.title("Proyecto Sudoku")
separador()

imagen_subida = st.file_uploader(
    "Sube una imagen de un sudoku",
    type=["jpg", "jpeg", "png"],
)

if imagen_subida is None:
    st.write("Sube una imagen para empezar.")
else:
    imagen = Image.open(imagen_subida)
    separador()
    titulo_seccion("Imagen subida")
    st.image(imagen, caption="Imagen subida", width="stretch")
    clave_imagen = obtener_clave_imagen(imagen_subida)

    if st.session_state.get("imagen_clave") != clave_imagen:
        limpiar_resultados_anteriores()
        st.session_state["imagen_clave"] = clave_imagen
        ruta_temporal = guardar_imagen_temporal(imagen_subida)

        try:
            with st.spinner("Detectando tablero y leyendo casillas..."):
                tablero_recortado, tablero_predicho, confianzas = leer_imagen(ruta_temporal)

            st.session_state["tablero_recortado"] = tablero_recortado
            st.session_state["tablero_predicho"] = tablero_predicho
            st.session_state["confianzas"] = confianzas

            with st.spinner("Resolviendo sudoku..."):
                tablero_resuelto, error_solucion = resolver_tablero_leido(tablero_predicho)

            st.session_state["tablero_resuelto"] = tablero_resuelto
            st.session_state["error_solucion"] = error_solucion

        except Exception as error:
            st.session_state["error_proceso"] = f"Ha ocurrido un error: {error}"

        finally:
            ruta_temporal.unlink(missing_ok=True)

    if "error_proceso" in st.session_state:
        st.error(st.session_state["error_proceso"])

    if "tablero_predicho" in st.session_state:
        try:
            tablero_recortado = st.session_state["tablero_recortado"]
            tablero_predicho = st.session_state["tablero_predicho"]
            confianzas = st.session_state["confianzas"]

            separador()
            titulo_seccion("Tablero detectado")
            tablero_rgb = cv2.cvtColor(tablero_recortado, cv2.COLOR_BGR2RGB)
            st.image(tablero_rgb, caption="Tablero detectado por YOLO", width="stretch")

            mostrar_tablero_leido_cnn(tablero_predicho)

            mostrar_diagnostico_lectura(tablero_predicho, confianzas)

            if st.session_state["error_solucion"] is not None:
                st.error(st.session_state["error_solucion"])
            else:
                separador()
                mostrar_sudoku_resuelto(
                    tablero_predicho,
                    st.session_state["tablero_resuelto"],
                )

        except Exception as error:
            st.error(f"Ha ocurrido un error: {error}")
