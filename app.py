# =============================================================================
# PROYECTO: ANÁLISIS DE DATOS DE CASINO / PÓKER
# ARCHIVO:  app.py  —  Dashboard Interactivo con Streamlit
# EJECUTAR: streamlit run app.py
# MATERIA:  Extracción de Conocimientos de Base de Datos
# =============================================================================

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import (mean_absolute_error, mean_squared_error,
                             r2_score, accuracy_score, confusion_matrix,
                             classification_report)
import os

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURACIÓN DE LA PÁGINA
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Análisis de Casino - Póker",
    page_icon="♠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─────────────────────────────────────────────────────────────────────────────
# 1. CARGA DE DATOS (con caché para mayor eficiencia)
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_data
def cargar_datos():
    ruta = os.path.join("data", "clean", "poker_manos_clean.csv")
    df = pd.read_csv(ruta)
    return df

df = cargar_datos()

# ─────────────────────────────────────────────────────────────────────────────
# 2. ENTRENAMIENTO DE MODELOS (con caché)
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_data
def entrenar_modelos(df):
    # --- Regresión lineal ---
    X_reg = df[["carta1","carta2","carta3","carta4","carta5","cartas_distintas","clase_mano"]]
    y_reg = df["carta_maxima"]
    X_r_train, X_r_test, y_r_train, y_r_test = train_test_split(
        X_reg, y_reg, test_size=0.3, random_state=42)
    modelo_reg = LinearRegression()
    modelo_reg.fit(X_r_train, y_r_train)
    pred_reg  = modelo_reg.predict(X_r_test)
    mae  = mean_absolute_error(y_r_test, pred_reg)
    rmse = np.sqrt(mean_squared_error(y_r_test, pred_reg))
    r2   = r2_score(y_r_test, pred_reg)

    # --- Árbol de decisión ---
    X_cls = df[["carta1","carta2","carta3","carta4","carta5",
                "carta_maxima","cartas_distintas","suma_cartas"]]
    y_cls = df["gano"]
    X_c_train, X_c_test, y_c_train, y_c_test = train_test_split(
        X_cls, y_cls, test_size=0.3, random_state=42)
    modelo_arbol = DecisionTreeClassifier(max_depth=5, random_state=42)
    modelo_arbol.fit(X_c_train, y_c_train)
    pred_cls  = modelo_arbol.predict(X_c_test)
    exactitud = accuracy_score(y_c_test, pred_cls)
    mc        = confusion_matrix(y_c_test, pred_cls)
    reporte   = classification_report(y_c_test, pred_cls,
                   target_names=["Perdio","Gano"], output_dict=True)

    return {
        "modelo_reg": modelo_reg, "X_reg": X_reg, "y_reg": y_reg,
        "X_r_test": X_r_test, "y_r_test": y_r_test, "pred_reg": pred_reg,
        "mae": mae, "rmse": rmse, "r2": r2,
        "modelo_arbol": modelo_arbol, "X_cls": X_cls, "y_cls": y_cls,
        "X_c_test": X_c_test, "y_c_test": y_c_test, "pred_cls": pred_cls,
        "exactitud": exactitud, "mc": mc, "reporte": reporte
    }

modelos = entrenar_modelos(df)

# ─────────────────────────────────────────────────────────────────────────────
# 3. SIDEBAR — FILTROS
# ─────────────────────────────────────────────────────────────────────────────
st.sidebar.title("Filtros del Dataset")
st.sidebar.markdown("---")

# Filtro: tipo de mano
manos_disponibles = sorted(df["nombre_mano"].unique().tolist())
manos_seleccionadas = st.sidebar.multiselect(
    "Tipos de mano a mostrar",
    options=manos_disponibles,
    default=manos_disponibles
)

# Filtro: rango de suma de cartas
min_suma = int(df["suma_cartas"].min())
max_suma = int(df["suma_cartas"].max())
rango_suma = st.sidebar.slider(
    "Rango de suma de cartas",
    min_value=min_suma, max_value=max_suma,
    value=(min_suma, max_suma)
)

# Filtro: sólo ganadoras / sólo perdedoras
filtro_gano = st.sidebar.radio(
    "Mostrar manos:",
    options=["Todas", "Sólo ganadoras", "Sólo perdedoras"],
    index=0
)

# Aplicar filtros
df_filtrado = df[
    (df["nombre_mano"].isin(manos_seleccionadas)) &
    (df["suma_cartas"].between(*rango_suma))
]
if filtro_gano == "Sólo ganadoras":
    df_filtrado = df_filtrado[df_filtrado["gano"] == 1]
elif filtro_gano == "Sólo perdedoras":
    df_filtrado = df_filtrado[df_filtrado["gano"] == 0]

st.sidebar.markdown("---")
st.sidebar.markdown(f"**Registros visibles:** {len(df_filtrado):,} / {len(df):,}")

# ─────────────────────────────────────────────────────────────────────────────
# 4. ENCABEZADO PRINCIPAL
# ─────────────────────────────────────────────────────────────────────────────
st.title("Analisis de Casino - Dataset de Manos de Poker")
st.markdown("""
**Dataset:** UCI Poker Hand | **Registros:** 5,000 manos
""")

# ─────────────────────────────────────────────────────────────────────────────
# 5. KPIs PRINCIPALES
# ─────────────────────────────────────────────────────────────────────────────
col1, col2, col3, col4, col5 = st.columns(5)

with col1:
    st.metric("Total de registros", f"{len(df_filtrado):,}")
with col2:
    st.metric("% Manos ganadoras", f"{df_filtrado['gano'].mean()*100:.1f}%")
with col3:
    st.metric("Suma promedio", f"{df_filtrado['suma_cartas'].mean():.1f}")
with col4:
    st.metric("Carta máxima prom.", f"{df_filtrado['carta_maxima'].mean():.1f}")
with col5:
    st.metric("Tipos de mano", f"{df_filtrado['nombre_mano'].nunique()}")

st.markdown("---")

# ─────────────────────────────────────────────────────────────────────────────
# 6. PESTAÑAS PRINCIPALES
# ─────────────────────────────────────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs(["ETL / Datos", "Regresion Lineal", "Arbol de Decision"])

# ══════════════════════════════════════════════════════
# PESTAÑA 1: ETL / DATOS
# ══════════════════════════════════════════════════════
with tab1:
    st.header("Exploración del Dataset (Post-ETL)")

    # Vista de datos
    st.subheader("Muestra del dataset limpio")
    st.dataframe(df_filtrado.head(50), use_container_width=True)

    col_a, col_b = st.columns(2)

    with col_a:
        st.subheader("Distribución de tipos de mano")
        conteo = df_filtrado["nombre_mano"].value_counts().reset_index()
        conteo.columns = ["Tipo de mano", "Frecuencia"]
        fig1, ax1 = plt.subplots(figsize=(7, 4))
        colores = plt.cm.Set2.colors
        ax1.bar(conteo["Tipo de mano"], conteo["Frecuencia"],
                color=colores[:len(conteo)])
        ax1.set_title("Frecuencia de Tipos de Mano", fontsize=12)
        ax1.set_xlabel("Tipo de mano"); ax1.set_ylabel("Cantidad")
        ax1.tick_params(axis="x", rotation=40)
        ax1.grid(axis="y", alpha=0.3)
        plt.tight_layout()
        st.pyplot(fig1)
        plt.close()

    with col_b:
        st.subheader("Proporción Ganó / Perdió")
        conteo_gano = df_filtrado["gano"].value_counts().reindex([0, 1], fill_value=0)
        fig2, ax2 = plt.subplots(figsize=(5, 5))
        if conteo_gano.sum() > 0:
            ax2.pie(conteo_gano, labels=["Perdió", "Ganó"],
                    autopct="%1.1f%%", colors=["#d62728", "#2ca02c"],
                    startangle=140, wedgeprops={"edgecolor":"white"})
        else:
            ax2.text(0.5, 0.5, "Sin datos para mostrar",
                     ha="center", va="center", transform=ax2.transAxes)
            ax2.axis("off")
        ax2.set_title("Ganó vs Perdió", fontsize=12)
        plt.tight_layout()
        st.pyplot(fig2)
        plt.close()

    st.subheader("Histograma: Suma de Cartas")
    fig3, ax3 = plt.subplots(figsize=(9, 4))
    ax3.hist(df_filtrado["suma_cartas"], bins=25, color="#4C72B0",
             edgecolor="white", alpha=0.85)
    ax3.axvline(df_filtrado["suma_cartas"].mean(), color="red",
                linestyle="--", linewidth=2, label=f"Media = {df_filtrado['suma_cartas'].mean():.1f}")
    ax3.set_title("Distribución de la Suma de Cartas", fontsize=12)
    ax3.set_xlabel("Suma de cartas"); ax3.set_ylabel("Frecuencia")
    ax3.legend(); ax3.grid(alpha=0.3)
    plt.tight_layout()
    st.pyplot(fig3)
    plt.close()

    st.subheader("Promedios por tipo de mano")
    tabla = df_filtrado.groupby("nombre_mano")[["suma_cartas","carta_maxima","cartas_distintas"]].mean().round(2)
    st.dataframe(tabla, use_container_width=True)

# ══════════════════════════════════════════════════════
# PESTAÑA 2: REGRESIÓN LINEAL
# ══════════════════════════════════════════════════════
with tab2:
    st.header("Regresión Lineal — Predicción de Carta Máxima")
    st.info("**Variable objetivo:** `carta_maxima` — El modelo predice el valor de la carta más alta de la mano usando las 5 cartas individuales, `cartas_distintas` y `clase_mano` como predictoras.")

    # Métricas
    c1, c2, c3 = st.columns(3)
    with c1: st.metric("MAE", f"{modelos['mae']:.4f}")
    with c2: st.metric("RMSE", f"{modelos['rmse']:.4f}")
    with c3: st.metric("R²", f"{modelos['r2']:.4f} ({modelos['r2']*100:.2f}%)")

    col_reg1, col_reg2 = st.columns(2)

    with col_reg1:
        st.subheader("Valores reales vs predichos")
        fig4, ax4 = plt.subplots(figsize=(6, 5))
        ax4.scatter(modelos["y_r_test"], modelos["pred_reg"],
                    alpha=0.3, color="#4C72B0", s=15)
        lim_min = min(modelos["y_r_test"].min(), modelos["pred_reg"].min())
        lim_max = max(modelos["y_r_test"].max(), modelos["pred_reg"].max())
        ax4.plot([lim_min, lim_max],[lim_min, lim_max],"r--", linewidth=2, label="Línea ideal")
        ax4.set_title("Reales vs Predichos (Carta Máxima)", fontsize=11)
        ax4.set_xlabel("Real"); ax4.set_ylabel("Predicho")
        ax4.legend(); ax4.grid(alpha=0.3)
        plt.tight_layout(); st.pyplot(fig4); plt.close()

    with col_reg2:
        st.subheader("Coeficientes del modelo")
        coefs = pd.Series(modelos["modelo_reg"].coef_,
                          index=["carta1","carta2","carta3","carta4","carta5","cartas_distintas","clase_mano"])
        fig5, ax5 = plt.subplots(figsize=(6, 4))
        colores_c = ["#2ca02c" if c > 0 else "#d62728" for c in coefs]
        ax5.barh(coefs.index, coefs.values, color=colores_c)
        ax5.axvline(0, color="black", linewidth=0.8)
        ax5.set_title("Importancia de Coeficientes", fontsize=11)
        ax5.grid(axis="x", alpha=0.3)
        plt.tight_layout(); st.pyplot(fig5); plt.close()

    # Predictor interactivo
    st.markdown("---")
    st.subheader("Prediccion Interactiva - Regresion")
    c_p1, c_p2, c_p3, c_p4 = st.columns(4)
    with c_p1:
        p_c1 = st.slider("Carta 1", 1, 13, 7)
        p_c2 = st.slider("Carta 2", 1, 13, 9)
    with c_p2:
        p_c3 = st.slider("Carta 3", 1, 13, 3)
        p_c4 = st.slider("Carta 4", 1, 13, 11)
    with c_p3:
        p_c5 = st.slider("Carta 5", 1, 13, 6)
        p_dist = st.slider("Cartas distintas", 1, 5, 5)
    with c_p4:
        p_clase = st.slider("Clase de mano", 0, 9, 0)

    if st.button("Predecir Carta Máxima", key="btn_reg"):
        entrada = pd.DataFrame({
            "carta1":[p_c1],"carta2":[p_c2],"carta3":[p_c3],
            "carta4":[p_c4],"carta5":[p_c5],
            "cartas_distintas":[p_dist],"clase_mano":[p_clase]
        })
        pred_val = modelos["modelo_reg"].predict(entrada)[0]
        real_max = max(p_c1, p_c2, p_c3, p_c4, p_c5)
        st.success(f"**Carta máxima real:** {real_max} | **Predicción del modelo:** {pred_val:.2f} | **Error:** {abs(real_max - pred_val):.2f}")

# ══════════════════════════════════════════════════════
# PESTAÑA 3: ÁRBOL DE DECISIÓN
# ══════════════════════════════════════════════════════
with tab3:
    st.header("Árbol de Decisión — ¿La mano ganará o perderá?")
    st.info("**Variable objetivo:** `gano` (0=Perdió, 1=Ganó). Un par o mejor = ganadora. Carta alta sola = perdedora.")

    # Métricas
    c1, c2, c3 = st.columns(3)
    with c1: st.metric("Exactitud", f"{modelos['exactitud']*100:.2f}%")
    with c2: st.metric("Precisión (Ganó)", f"{modelos['reporte']['Gano']['precision']*100:.2f}%")
    with c3: st.metric("Recall (Ganó)", f"{modelos['reporte']['Gano']['recall']*100:.2f}%")

    col_arb1, col_arb2 = st.columns(2)

    with col_arb1:
        st.subheader("Importancia de variables")
        importancias = pd.Series(
            modelos["modelo_arbol"].feature_importances_,
            index=["carta1","carta2","carta3","carta4","carta5",
                   "carta_maxima","cartas_distintas","suma_cartas"]
        ).sort_values(ascending=True)
        fig6, ax6 = plt.subplots(figsize=(6, 4))
        colores_imp = plt.cm.RdYlGn(np.linspace(0.3, 0.9, len(importancias)))
        ax6.barh(importancias.index, importancias.values, color=colores_imp)
        ax6.set_title("Importancia de Variables (Gini)", fontsize=11)
        ax6.set_xlabel("Importancia")
        ax6.grid(axis="x", alpha=0.3)
        plt.tight_layout(); st.pyplot(fig6); plt.close()

    with col_arb2:
        st.subheader("Matriz de confusión")
        mc = modelos["mc"]
        fig7, ax7 = plt.subplots(figsize=(5, 4))
        ax7.imshow(mc, cmap="Blues")
        ax7.set_xticks([0,1]); ax7.set_yticks([0,1])
        ax7.set_xticklabels(["Pred: Perdió","Pred: Ganó"])
        ax7.set_yticklabels(["Real: Perdió","Real: Ganó"])
        for i in range(2):
            for j in range(2):
                ax7.text(j, i, str(mc[i,j]), ha="center", va="center",
                         fontsize=16, color="white" if mc[i,j] > mc.max()/2 else "black")
        ax7.set_title("Matriz de Confusión", fontsize=11)
        plt.tight_layout(); st.pyplot(fig7); plt.close()

    st.subheader("Árbol de decisión (max_depth=5)")
    fig8, ax8 = plt.subplots(figsize=(18, 7))
    plot_tree(modelos["modelo_arbol"],
              feature_names=["carta1","carta2","carta3","carta4","carta5",
                             "carta_maxima","cartas_distintas","suma_cartas"],
              class_names=["Perdio","Gano"],
              filled=True, rounded=True, fontsize=8, ax=ax8,
              impurity=False, proportion=True)
    plt.tight_layout(); st.pyplot(fig8); plt.close()

    # Predictor interactivo
    st.markdown("---")
    st.subheader("Prediccion Interactiva - Arbol de Decision")
    c_a1, c_a2, c_a3 = st.columns(3)
    with c_a1:
        a_c1 = st.slider("Carta 1", 1, 13, 5, key="a1")
        a_c2 = st.slider("Carta 2", 1, 13, 5, key="a2")
        a_c3 = st.slider("Carta 3", 1, 13, 5, key="a3")
    with c_a2:
        a_c4 = st.slider("Carta 4", 1, 13, 6, key="a4")
        a_c5 = st.slider("Carta 5", 1, 13, 6, key="a5")
    with c_a3:
        st.markdown("*La carta máxima, distintas y suma se calculan automáticamente.*")

    if st.button("Predecir si ganará", key="btn_arb"):
        a_max  = max(a_c1, a_c2, a_c3, a_c4, a_c5)
        a_dist = len(set([a_c1, a_c2, a_c3, a_c4, a_c5]))
        a_suma = a_c1 + a_c2 + a_c3 + a_c4 + a_c5
        entrada_arb = pd.DataFrame({
            "carta1":[a_c1],"carta2":[a_c2],"carta3":[a_c3],
            "carta4":[a_c4],"carta5":[a_c5],
            "carta_maxima":[a_max],"cartas_distintas":[a_dist],"suma_cartas":[a_suma]
        })
        pred_clase = modelos["modelo_arbol"].predict(entrada_arb)[0]
        prob_gana  = modelos["modelo_arbol"].predict_proba(entrada_arb)[0][1] * 100

        if pred_clase == 1:
            st.success(f"El jugador GANARA -- Probabilidad: {prob_gana:.1f}%")
        else:
            st.error(f"El jugador NO GANARA -- Probabilidad de ganar: {prob_gana:.1f}%")
        st.markdown(f"*Cartas distintas: {a_dist} | Suma: {a_suma} | Carta máxima: {a_max}*")
