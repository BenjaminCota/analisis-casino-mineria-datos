# Analisis de Casino - Manos de Poker

> **Materia:** Extraccion de Conocimientos de Base de Datos
> **Dataset:** UCI Poker Hand (5,000 manos de poker reales)
> **Universidad:** JABAKH

---

## Descripcion del Proyecto

Proyecto de mineria de datos que aplica el proceso ETL completo, regresion lineal y arbol de decision sobre el dataset publico **UCI Poker Hand**, el cual contiene informacion de manos de poker clasificadas por tipo. Se desarrollo adicionalmente un dashboard interactivo con Streamlit y un reporte academico en Word.

---

## Instalacion

```bash
# 1. Clonar el repositorio
git clone https://github.com/[USUARIO]/analisis-casino-mineria-datos.git
cd analisis-casino-mineria-datos

# 2. Crear entorno virtual (recomendado)
python -m venv venv
venv\Scripts\activate   # Windows

# 3. Instalar dependencias
pip install -r requirements.txt
```

---

## Ejecucion

### Fase 1 - ETL
```bash
cd src
python 01_etl.py
```
Genera: `data/raw/poker_manos_raw.csv`, `data/clean/poker_manos_clean.csv`, `data/clean/casino.db`, `data/clean/resultados_casino.xlsx` y 3 graficas PNG.

### Fase 2 - Regresion Lineal
```bash
python 02_regresion_lineal.py
```
Genera: 3 graficas de regresion en `reports/figures/`.

### Fase 3 - Arbol de Decision
```bash
python 03_arbol_decision.py
```
Genera: 4 graficas del arbol en `reports/figures/`.

### Fase 4 - Dashboard Streamlit
```bash
cd ..
streamlit run app.py
```
Abre el navegador en `http://localhost:8501`

### Fase 5 - Reporte Word
```bash
cd reports
python generar_word.py
```
Genera: `reports/reporte_casino.docx`

---

## Estructura del Proyecto

```
proyecto-casino/
├── data/
│   ├── raw/              <- Dataset original descargado de UCI
│   └── clean/            <- CSV limpio, SQLite y Excel de resultados
├── src/
│   ├── 01_etl.py         <- Proceso ETL completo
│   ├── 02_regresion_lineal.py  <- Modelo de regresion
│   └── 03_arbol_decision.py    <- Arbol de decision
├── reports/
│   ├── figures/          <- Graficas PNG generadas
│   ├── generar_word.py   <- Generador del reporte Word
│   └── reporte_casino.docx     <- Reporte final
├── app.py                <- Dashboard Streamlit
├── requirements.txt
├── .gitignore
└── README.md
```

---

## Resultados Clave

| Modelo | Variable objetivo | Metrica principal |
|--------|------------------|-------------------|
| Regresion Lineal | `carta_maxima` | R2 = 0.4728 (47.28%) |
| Arbol de Decision | `gano` (binario) | Exactitud = 99.47% |

### Hallazgos destacados:
- **49.72%** de las manos tienen par o mejor (son ganadoras)
- La variable **`cartas_distintas`** explica el **99.18%** de la importancia en el arbol de decision
- Si una mano tiene 5 cartas distintas -> necesariamente es "Carta Alta" (pierde)
- Si una mano tiene 4 o menos cartas distintas -> existe al menos un par (gana)

---

## Dependencias

| Libreria | Uso |
|----------|-----|
| `pandas` | Manipulacion de datos |
| `numpy` | Calculos numericos |
| `matplotlib` | Visualizaciones estaticas |
| `scikit-learn` | Modelos de ML |
| `streamlit` | Dashboard interactivo |
| `openpyxl` | Exportar a Excel |
| `python-docx` | Generacion del reporte Word |

---

## Fuente del Dataset

- **UCI Poker Hand Dataset**: https://archive.ics.uci.edu/ml/datasets/Poker+Hand
- Sandri, R., y Zupan, B. (2010). *UCI Machine Learning Repository*.

---

*Proyecto desarrollado con Python 3.x para la materia de Extraccion de Conocimientos de Base de Datos.*
