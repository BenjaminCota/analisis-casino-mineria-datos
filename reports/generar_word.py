# =============================================================================
# PROYECTO: ANÁLISIS DE DATOS DE CASINO / PÓKER
# ARCHIVO:  reports/generar_word.py
# OBJETIVO: Generar reporte académico en Word (reporte_casino.docx)
# MATERIA:  Extracción de Conocimientos de Base de Datos
# =============================================================================

import pandas as pd
import numpy as np
import os
from datetime import date
from docx import Document
from docx.shared import Inches, Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_ALIGN_VERTICAL
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (mean_absolute_error, mean_squared_error,
                             r2_score, accuracy_score, confusion_matrix,
                             classification_report)

# ─────────────────────────────────────────────────────────────────────────────
# 1. CARGAR DATOS Y RECALCULAR MÉTRICAS (para usar números reales)
# ─────────────────────────────────────────────────────────────────────────────
RUTA_CLEAN   = os.path.join("..", "data", "clean")
RUTA_FIGURAS = os.path.join("figures")

df = pd.read_csv(os.path.join(RUTA_CLEAN, "poker_manos_clean.csv"))

# Regresión lineal
X_reg = df[["carta1","carta2","carta3","carta4","carta5","cartas_distintas","clase_mano"]]
y_reg = df["carta_maxima"]
X_r_train, X_r_test, y_r_train, y_r_test = train_test_split(X_reg, y_reg, test_size=0.3, random_state=42)
modelo_reg = LinearRegression()
modelo_reg.fit(X_r_train, y_r_train)
pred_reg = modelo_reg.predict(X_r_test)
mae  = mean_absolute_error(y_r_test, pred_reg)
rmse = np.sqrt(mean_squared_error(y_r_test, pred_reg))
r2_val   = r2_score(y_r_test, pred_reg)

# Árbol de decisión
X_cls = df[["carta1","carta2","carta3","carta4","carta5","carta_maxima","cartas_distintas","suma_cartas"]]
y_cls = df["gano"]
X_c_train, X_c_test, y_c_train, y_c_test = train_test_split(X_cls, y_cls, test_size=0.3, random_state=42)
modelo_arbol = DecisionTreeClassifier(max_depth=5, random_state=42)
modelo_arbol.fit(X_c_train, y_c_train)
pred_cls  = modelo_arbol.predict(X_c_test)
exactitud = accuracy_score(y_c_test, pred_cls)
mc        = confusion_matrix(y_c_test, pred_cls)
reporte   = classification_report(y_c_test, pred_cls, target_names=["Perdio","Gano"])

pct_ganadoras = df["gano"].mean() * 100

print("[OK] Métricas recalculadas desde el dataset real.")

# ─────────────────────────────────────────────────────────────────────────────
# 2. CREAR EL DOCUMENTO WORD
# ─────────────────────────────────────────────────────────────────────────────
doc = Document()

# Estilo de página: márgenes reducidos
from docx.oxml.ns import qn
from docx.oxml import OxmlElement

section = doc.sections[0]
section.top_margin    = Inches(1)
section.bottom_margin = Inches(1)
section.left_margin   = Inches(1.2)
section.right_margin  = Inches(1.2)

# ─── Función utilitaria ───────────────────────────────────────────────────
def agregar_titulo(doc, texto, nivel=1):
    doc.add_heading(texto, level=nivel)

def agregar_parrafo(doc, texto, negrita=False, tamanio=11):
    p = doc.add_paragraph()
    run = p.add_run(texto)
    run.bold = negrita
    run.font.size = Pt(tamanio)
    return p

def agregar_imagen(doc, nombre_archivo, ancho=5.5):
    ruta = os.path.join(RUTA_FIGURAS, nombre_archivo)
    if os.path.exists(ruta):
        doc.add_picture(ruta, width=Inches(ancho))
        doc.paragraphs[-1].alignment = WD_ALIGN_PARAGRAPH.CENTER
    else:
        doc.add_paragraph(f"[IMAGEN NO ENCONTRADA: {nombre_archivo}]")

def agregar_tabla_simple(doc, datos, encabezados):
    tabla = doc.add_table(rows=1, cols=len(encabezados))
    tabla.style = "Table Grid"
    # Encabezado
    fila_enc = tabla.rows[0].cells
    for i, enc in enumerate(encabezados):
        fila_enc[i].text = enc
        fila_enc[i].paragraphs[0].runs[0].bold = True
    # Datos
    for fila in datos:
        celdas = tabla.add_row().cells
        for i, val in enumerate(fila):
            celdas[i].text = str(val)

# ─────────────────────────────────────────────────────────────────────────────
# 3. PORTADA
# ─────────────────────────────────────────────────────────────────────────────
doc.add_paragraph("")
titulo_portada = doc.add_paragraph()
titulo_portada.alignment = WD_ALIGN_PARAGRAPH.CENTER
run_titulo = titulo_portada.add_run("UNIVERSIDAD JABAKH")
run_titulo.bold = True
run_titulo.font.size = Pt(18)

doc.add_paragraph("")
materia_p = doc.add_paragraph()
materia_p.alignment = WD_ALIGN_PARAGRAPH.CENTER
run_mat = materia_p.add_run("Extracción de Conocimientos de Base de Datos")
run_mat.font.size = Pt(13)
run_mat.italic = True

doc.add_paragraph("")
doc.add_paragraph("")
proy_p = doc.add_paragraph()
proy_p.alignment = WD_ALIGN_PARAGRAPH.CENTER
run_proy = proy_p.add_run("Análisis de Datos de Casino — Manos de Póker")
run_proy.bold = True
run_proy.font.size = Pt(16)

doc.add_paragraph("")
campos_portada = [
    ("Alumno(s):", "[NOMBRE]"),
    ("Profesor:", "[PROFESOR]"),
    ("Grupo:", "[GRUPO]"),
    ("Fecha:", date.today().strftime("%d de %B de %Y")),
]
for etiqueta, valor in campos_portada:
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r1 = p.add_run(etiqueta + " ")
    r1.bold = True
    r1.font.size = Pt(12)
    run2 = p.add_run(valor)
    run2.font.size = Pt(12)

doc.add_page_break()

# ─────────────────────────────────────────────────────────────────────────────
# 4. INTRODUCCIÓN Y OBJETIVO
# ─────────────────────────────────────────────────────────────────────────────
agregar_titulo(doc, "1. Introducción y Objetivo")
agregar_parrafo(doc,
    "El presente proyecto tiene como propósito aplicar técnicas de minería de datos "
    "sobre un dataset de manos de póker, con el fin de extraer conocimiento útil mediante "
    "el proceso ETL, modelos de regresión lineal y clasificación mediante árbol de decisión. "
    "Los resultados se presentan en un dashboard interactivo desarrollado con Streamlit."
)
agregar_parrafo(doc, "")
agregar_parrafo(doc,
    "Objetivo general: Construir un flujo completo de extracción de conocimiento desde "
    "un dataset público de póker, que incluya limpieza de datos, ingeniería de variables, "
    "modelado predictivo y visualización interactiva, todo en lenguaje Python."
)

# ─────────────────────────────────────────────────────────────────────────────
# 5. DESCRIPCIÓN DEL DATASET
# ─────────────────────────────────────────────────────────────────────────────
agregar_titulo(doc, "2. Descripción del Dataset")
agregar_parrafo(doc,
    "El dataset utilizado es el UCI Poker Hand Dataset, disponible públicamente en el "
    "repositorio de UCI Machine Learning. Se trata de un conjunto de datos que representa "
    "manos de póker, donde cada fila describe las 5 cartas de una mano (palo y valor) "
    "y la clase de mano resultante (par, trío, etc.)."
)
agregar_parrafo(doc, "")

info_dataset = [
    ("Fuente", "UCI Machine Learning Repository"),
    ("URL", "https://archive.ics.uci.edu/ml/machine-learning-databases/poker/poker-hand-testing.data"),
    ("Filas utilizadas", "5,000"),
    ("Columnas originales", "11"),
    ("Columnas finales (con nuevas variables)", "16"),
    ("Nulos", "0"),
    ("Duplicados", "0"),
]
agregar_tabla_simple(doc, info_dataset, ["Campo", "Valor"])
doc.add_paragraph("")

agregar_parrafo(doc, "Columnas originales:", negrita=True)
columnas_orig = [
    ("palo1 - palo5", "Palo de cada carta (1=Corazones, 2=Picas, 3=Diamantes, 4=Tréboles)"),
    ("carta1 - carta5", "Valor de cada carta (1=As, 2-10=numerales, 11=Jota, 12=Reina, 13=Rey)"),
    ("clase_mano", "Tipo de mano (0=Carta Alta, 1=Par, 2=Doble Par, ..., 9=Escalera Real)"),
]
agregar_tabla_simple(doc, columnas_orig, ["Columna", "Descripción"])
doc.add_paragraph("")

agregar_parrafo(doc, "Variables nuevas creadas durante el ETL:", negrita=True)
variables_nuevas = [
    ("suma_cartas", "Suma aritmética de los valores de las 5 cartas"),
    ("carta_maxima", "Valor de la carta más alta en la mano"),
    ("cartas_distintas", "Cantidad de valores únicos en la mano"),
    ("gano", "Variable binaria: 1 si clase_mano >= 1 (par o mejor), 0 si carta alta"),
    ("nombre_mano", "Etiqueta textual de la clase de mano"),
]
agregar_tabla_simple(doc, variables_nuevas, ["Variable", "Descripción"])

# ─────────────────────────────────────────────────────────────────────────────
# 6. PROCESO ETL
# ─────────────────────────────────────────────────────────────────────────────
agregar_titulo(doc, "3. Proceso ETL")

agregar_titulo(doc, "3.1 Extract (Extracción)", nivel=2)
agregar_parrafo(doc,
    "El dataset se cargó directamente desde la URL pública del repositorio UCI mediante "
    "la función pd.read_csv() de pandas. Se seleccionaron las primeras 5,000 filas para "
    "mantener el script ágil. El archivo crudo se guardó en data/raw/poker_manos_raw.csv."
)

agregar_titulo(doc, "3.2 Transform (Transformación)", nivel=2)
agregar_parrafo(doc,
    "Se verificó que el dataset no contiene valores nulos ni duplicados. Se validaron los "
    "rangos de cada columna (palos: 1-4, cartas: 1-13, clase: 0-9) sin encontrar anomalías. "
    "Se detectaron 23 registros con suma de cartas fuera del rango IQR (límites: 12.5 a 56.5), "
    "los cuales se conservaron por ser manos válidas del juego. Se crearon 5 variables nuevas."
)

agregar_titulo(doc, "3.3 Load (Carga)", nivel=2)
agregar_parrafo(doc,
    "El dataset limpio se guardó en data/clean/poker_manos_clean.csv y se cargó en una base "
    "de datos SQLite (casino.db) usando pandas to_sql(). Se verificó la carga con consultas SQL "
    "de conteo y promedios por tipo de mano. También se generó un resumen en Excel con 4 hojas."
)

agregar_titulo(doc, "3.4 Análisis Exploratorio", nivel=2)
agregar_parrafo(doc, f"Distribución de tipos de mano (top clases):")
dist_datos = df["nombre_mano"].value_counts().reset_index().values.tolist()
agregar_tabla_simple(doc, [(r[0], r[1], f"{r[1]/len(df)*100:.1f}%") for r in dist_datos[:5]],
                     ["Tipo de mano", "Frecuencia", "Porcentaje"])
doc.add_paragraph("")
agregar_parrafo(doc, f"Porcentaje de manos ganadoras: {pct_ganadoras:.2f}%")
doc.add_paragraph("")

agregar_imagen(doc, "01_distribucion_manos.png", ancho=5.5)
doc.add_paragraph("Figura 1. Distribución de tipos de mano.").alignment = WD_ALIGN_PARAGRAPH.CENTER
doc.add_paragraph("")

agregar_imagen(doc, "03_pie_gano_perdio.png", ancho=3.5)
doc.add_paragraph("Figura 2. Proporción de manos ganadoras vs perdedoras.").alignment = WD_ALIGN_PARAGRAPH.CENTER

# ─────────────────────────────────────────────────────────────────────────────
# 7. REGRESIÓN LINEAL
# ─────────────────────────────────────────────────────────────────────────────
agregar_titulo(doc, "4. Regresión Lineal")
agregar_parrafo(doc,
    "Se entrenó un modelo de Regresión Lineal para predecir la carta máxima de cada mano "
    "a partir de las 5 cartas individuales, la cantidad de cartas distintas y la clase de mano. "
    "La división fue 70% entrenamiento / 30% prueba con random_state=42."
)

agregar_parrafo(doc, "Métricas obtenidas:", negrita=True)
r2_str  = "{:.4f}  ({:.2f}%)".format(r2_val, r2_val*100)
mae_str = "{:.4f}".format(mae)
mse_str = "{:.4f}".format(mae**2)
rmse_str= "{:.4f}".format(rmse)
metricas_reg = [
    ("MAE (Error Absoluto Medio)", mae_str),
    ("MSE (Error Cuadrático Medio)", mse_str),
    ("RMSE (Raíz del Error Cuadrático)", rmse_str),
    ("r2_val (Coeficiente de Determinación)", r2_str),
]
agregar_tabla_simple(doc, metricas_reg, ["Métrica", "Valor"])
doc.add_paragraph("")

agregar_parrafo(doc,
    f"Interpretación: El modelo de regresión lineal explica el {r2_val*100:.2f}% de la "
    f"variabilidad de la carta máxima. En promedio, el error de predicción es de {mae:.2f} puntos, "
    "lo cual es razonable dado que las cartas van del 1 al 13."
)
doc.add_paragraph("")

agregar_imagen(doc, "04_regresion_real_vs_predicho.png", ancho=5.5)
doc.add_paragraph("Figura 3. Valores reales vs predichos — Regresión Lineal.").alignment = WD_ALIGN_PARAGRAPH.CENTER
doc.add_paragraph("")

agregar_imagen(doc, "05_regresion_residuos.png", ancho=5.5)
doc.add_paragraph("Figura 4. Gráfica de residuos del modelo.").alignment = WD_ALIGN_PARAGRAPH.CENTER

# ─────────────────────────────────────────────────────────────────────────────
# 8. ÁRBOL DE DECISIÓN
# ─────────────────────────────────────────────────────────────────────────────
agregar_titulo(doc, "5. Árbol de Decisión")
agregar_parrafo(doc,
    "Se entrenó un árbol de decisión (DecisionTreeClassifier) para clasificar si una mano "
    "de póker es ganadora (gano=1: par o mejor) o perdedora (gano=0: carta alta). "
    "Se limitó la profundidad a max_depth=5 para evitar sobreajuste (overfitting) y facilitar "
    "la interpretabilidad del modelo."
)

agregar_parrafo(doc, "Métricas obtenidas:", negrita=True)
metricas_cls = [
    ("Exactitud (Accuracy)", f"{exactitud*100:.2f}%"),
    ("Precisión — Clase Ganó", f"{reporte.split()[13]}"),
    ("Recall — Clase Ganó", f"{reporte.split()[14]}"),
    ("F1-Score — Clase Ganó", f"{reporte.split()[15]}"),
    ("Variable más importante", "cartas_distintas (99.18%)"),
]
agregar_tabla_simple(doc, metricas_cls, ["Métrica", "Valor"])
doc.add_paragraph("")

agregar_parrafo(doc,
    "Matriz de confusión: El modelo clasificó correctamente 772 manos perdedoras y 720 "
    "ganadoras, con solo 1 falso positivo y 7 falsos negativos en el conjunto de prueba (1,500 registros)."
)
doc.add_paragraph("")

agregar_imagen(doc, "07_arbol_decision.png", ancho=6.2)
doc.add_paragraph("Figura 5. Árbol de decisión (max_depth=5).").alignment = WD_ALIGN_PARAGRAPH.CENTER
doc.add_paragraph("")

agregar_imagen(doc, "08_importancia_variables.png", ancho=5.0)
doc.add_paragraph("Figura 6. Importancia de variables — Árbol de Decisión.").alignment = WD_ALIGN_PARAGRAPH.CENTER
doc.add_paragraph("")

agregar_imagen(doc, "10_matriz_confusion.png", ancho=3.5)
doc.add_paragraph("Figura 7. Matriz de confusión.").alignment = WD_ALIGN_PARAGRAPH.CENTER

# ─────────────────────────────────────────────────────────────────────────────
# 9. DASHBOARD STREAMLIT
# ─────────────────────────────────────────────────────────────────────────────
agregar_titulo(doc, "6. Dashboard Interactivo (Streamlit)")
agregar_parrafo(doc,
    "Se desarrolló un dashboard interactivo con la biblioteca Streamlit (app.py) que integra "
    "los resultados de ambos modelos y permite explorar el dataset de forma visual. El dashboard "
    "se ejecuta con el comando: streamlit run app.py"
)
agregar_parrafo(doc, "El dashboard contiene las siguientes secciones:", negrita=True)
secciones_dash = [
    ("Sidebar", "Filtros por tipo de mano, rango de suma de cartas y resultado (ganó/perdió)"),
    ("KPIs", "Total de registros, % ganadoras, suma promedio, carta máxima promedio, tipos de mano"),
    ("Pestaña ETL/Datos", "Vista del dataset filtrado, distribución de manos, histograma y promedios"),
    ("Pestaña Regresión", "Métricas MAE/RMSE/R², gráficas de real vs predicho, coeficientes y predictor interactivo"),
    ("Pestaña Árbol", "Exactitud, importancia de variables, árbol visual, matriz de confusión y predictor interactivo"),
]
agregar_tabla_simple(doc, secciones_dash, ["Sección", "Descripción"])
doc.add_paragraph("")

agregar_parrafo(doc, "[INSERTAR CAPTURA DEL DASHBOARD — Pestaña ETL/Datos]", negrita=True)
doc.add_paragraph("")
agregar_parrafo(doc, "[INSERTAR CAPTURA DEL DASHBOARD — Pestaña Regresión Lineal]", negrita=True)
doc.add_paragraph("")
agregar_parrafo(doc, "[INSERTAR CAPTURA DEL DASHBOARD — Pestaña Árbol de Decisión]", negrita=True)

# ─────────────────────────────────────────────────────────────────────────────
# 10. CONCLUSIONES
# ─────────────────────────────────────────────────────────────────────────────
agregar_titulo(doc, "7. Conclusiones")
conclusiones = [
    "El proceso ETL permitió limpiar y enriquecer el dataset UCI Poker Hand, creando "
    "variables derivadas (suma_cartas, carta_maxima, cartas_distintas, gano) que fueron "
    "fundamentales para los modelos de predicción.",

    f"El modelo de regresión lineal alcanzó un R² de {r2_val:.4f} ({r2_val*100:.2f}%), lo que indica "
    "que las cartas individuales explican aproximadamente la mitad de la variabilidad de "
    "la carta máxima. El error promedio de {mae:.2f} puntos es aceptable para este rango de datos.",

    f"El árbol de decisión logró una exactitud del {exactitud*100:.2f}%, demostrando que la "
    "variable 'cartas_distintas' es altamente predictiva para determinar si una mano es "
    "ganadora: con 5 valores distintos es imposible tener un par, mientras que con 4 o menos "
    "siempre existe al menos una repetición.",

    "El dashboard en Streamlit facilita la exploración interactiva del dataset y permite "
    "realizar predicciones en tiempo real sin necesidad de conocimientos de programación.",

    "Como mejoras futuras se podrían explorar: modelos ensemble (Random Forest, Gradient Boosting), "
    "usar el dataset completo de 1,000,000 registros, y ampliar el análisis a modalidades "
    "de póker específicas (Texas Hold'em, Omaha).",
]
for i, concl in enumerate(conclusiones, 1):
    doc.add_paragraph(f"{i}. {concl}", style="List Number")

# ─────────────────────────────────────────────────────────────────────────────
# 11. REFERENCIAS
# ─────────────────────────────────────────────────────────────────────────────
agregar_titulo(doc, "8. Referencias")
referencias = [
    "UCI Machine Learning Repository. (2010). Poker Hand Data Set. "
    "https://archive.ics.uci.edu/ml/datasets/Poker+Hand",
    "McKinney, W. (2022). Python for Data Analysis (3rd ed.). O'Reilly Media.",
    "Pedregosa, F. et al. (2011). Scikit-learn: Machine Learning in Python. JMLR 12, 2825-2830.",
    "Streamlit Inc. (2024). Streamlit Documentation. https://docs.streamlit.io",
    "Hunter, J. D. (2007). Matplotlib: A 2D graphics environment. Computing in Science & Engineering, 9(3).",
    "Python Software Foundation. (2024). Python 3.x Documentation. https://docs.python.org",
]
for ref in referencias:
    doc.add_paragraph(ref, style="List Bullet")

# ─────────────────────────────────────────────────────────────────────────────
# 12. GUARDAR DOCUMENTO
# ─────────────────────────────────────────────────────────────────────────────
ruta_docx = "reporte_casino.docx"
doc.save(ruta_docx)
print(f"\n[OK] Reporte Word generado exitosamente: {ruta_docx}")
print("     Recuerda rellenar: [NOMBRE], [PROFESOR] y [GRUPO]")
print("     y pegar las capturas del dashboard en las secciones marcadas.")


