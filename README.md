# 🃏 Análisis de Casino — Manos de Póker

> **Materia:** Extracción de Conocimientos de Base de Datos  
> **Dataset:** UCI Poker Hand (5,000 manos de póker reales)  
> **Universidad:** JABAKH

---

## 📋 Descripción del Proyecto

Proyecto de minería de datos que aplica el proceso ETL completo, regresión lineal y árbol de decisión sobre el dataset público **UCI Poker Hand**, el cual contiene información de manos de póker clasificadas por tipo. Se desarrolló adicionalmente un dashboard interactivo con Streamlit y un reporte académico en Word.

---

## 🚀 Instalación

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

## ▶️ Ejecución

### Fase 1 — ETL
```bash
cd src
python 01_etl.py
```
Genera: `data/raw/poker_manos_raw.csv`, `data/clean/poker_manos_clean.csv`, `data/clean/casino.db`, `data/clean/resultados_casino.xlsx` y 3 gráficas PNG.

### Fase 2 — Regresión Lineal
```bash
python 02_regresion_lineal.py
```
Genera: 3 gráficas de regresión en `reports/figures/`.

### Fase 3 — Árbol de Decisión
```bash
python 03_arbol_decision.py
```
Genera: 4 gráficas del árbol en `reports/figures/`.

### Fase 4 — Dashboard Streamlit
```bash
cd ..
streamlit run app.py
```
Abre el navegador en `http://localhost:8501`

### Fase 5 — Reporte Word
```bash
cd reports
python generar_word.py
```
Genera: `reports/reporte_casino.docx`

---

## 📁 Estructura del Proyecto

```
proyecto-casino/
├── data/
│   ├── raw/              ← Dataset original descargado de UCI
│   └── clean/            ← CSV limpio, SQLite y Excel de resultados
├── src/
│   ├── 01_etl.py         ← Proceso ETL completo
│   ├── 02_regresion_lineal.py  ← Modelo de regresión
│   └── 03_arbol_decision.py    ← Árbol de decisión
├── reports/
│   ├── figures/          ← Gráficas PNG generadas
│   ├── generar_word.py   ← Generador del reporte Word
│   └── reporte_casino.docx     ← Reporte final
├── app.py                ← Dashboard Streamlit
├── requirements.txt
├── .gitignore
└── README.md
```

---

## 📊 Resultados Clave

| Modelo | Variable objetivo | Métrica principal |
|--------|------------------|-------------------|
| Regresión Lineal | `carta_maxima` | R² = 0.4728 (47.28%) |
| Árbol de Decisión | `gano` (binario) | Exactitud = 99.47% |

### Hallazgos destacados:
- **49.72%** de las manos tienen par o mejor (son ganadoras)
- La variable **`cartas_distintas`** explica el **99.18%** de la importancia en el árbol de decisión
- Si una mano tiene 5 cartas distintas → necesariamente es "Carta Alta" (pierde)
- Si una mano tiene 4 o menos cartas distintas → existe al menos un par (gana)

---

## 📦 Dependencias

| Librería | Uso |
|----------|-----|
| `pandas` | Manipulación de datos |
| `numpy` | Cálculos numéricos |
| `matplotlib` | Visualizaciones estáticas |
| `scikit-learn` | Modelos de ML |
| `streamlit` | Dashboard interactivo |
| `openpyxl` | Exportar a Excel |
| `python-docx` | Generación del reporte Word |

---

## 📚 Fuente del Dataset

- **UCI Poker Hand Dataset**: https://archive.ics.uci.edu/ml/datasets/Poker+Hand
- Sandri, R., & Zupan, B. (2010). *UCI Machine Learning Repository*.

---

*Proyecto desarrollado con Python 3.x para la materia de Extracción de Conocimientos de Base de Datos.*
