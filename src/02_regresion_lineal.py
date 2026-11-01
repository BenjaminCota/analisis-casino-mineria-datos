# =============================================================================
# PROYECTO: ANÁLISIS DE DATOS DE CASINO / PÓKER
# SCRIPT:   02_regresion_lineal.py
# OBJETIVO: Predecir la CARTA MÁXIMA de una mano usando regresión lineal
# MATERIA:  Extracción de Conocimientos de Base de Datos
# =============================================================================

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import os

# ─────────────────────────────────────────────────────────────────────────────
# 1. CONFIGURACIÓN DE RUTAS
# ─────────────────────────────────────────────────────────────────────────────
RUTA_FIGURAS = os.path.join("..", "reports", "figures")
RUTA_CLEAN   = os.path.join("..", "data", "clean")
os.makedirs(RUTA_FIGURAS, exist_ok=True)

# ─────────────────────────────────────────────────────────────────────────────
# 2. CARGAR EL DATASET LIMPIO
# ─────────────────────────────────────────────────────────────────────────────
print("=" * 65)
print("   REGRESIÓN LINEAL — Predicción de Carta Máxima")
print("=" * 65)

ruta_csv = os.path.join(RUTA_CLEAN, "poker_manos_clean.csv")
df = pd.read_csv(ruta_csv)

print(f"\nDataset cargado: {df.shape[0]:,} filas x {df.shape[1]} columnas")
print("\nPRIMEROS REGISTROS:")
print(df[["carta1","carta2","carta3","carta4","carta5","suma_cartas","carta_maxima","cartas_distintas"]].head().to_string())

# ─────────────────────────────────────────────────────────────────────────────
# 3. DEFINIR VARIABLES X (predictoras) y y (objetivo)
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "-" * 50)
print("VARIABLES DEL MODELO:")
print("-" * 50)

# Variables predictoras: las 5 cartas individuales y cartas distintas
X = df[["carta1", "carta2", "carta3", "carta4", "carta5", "cartas_distintas", "clase_mano"]]
y = df["carta_maxima"]   # Objetivo: carta más alta de la mano

print("Variables predictoras (X):")
for col in X.columns:
    print(f"  - {col}")
print(f"\nVariable objetivo (y): carta_maxima")
print(f"  Rango: [{y.min()} , {y.max()}]")
print(f"  Media: {y.mean():.2f}")

# ─────────────────────────────────────────────────────────────────────────────
# 4. DIVIDIR EN ENTRENAMIENTO Y PRUEBA (70% / 30%)
# ─────────────────────────────────────────────────────────────────────────────
X_entrenamiento, X_prueba, y_entrenamiento, y_prueba = train_test_split(
    X, y, test_size=0.3, random_state=42
)

print(f"\nDIVISIÓN DEL DATASET:")
print(f"  Entrenamiento: {X_entrenamiento.shape[0]:,} filas ({100*0.7:.0f}%)")
print(f"  Prueba:        {X_prueba.shape[0]:,} filas ({100*0.3:.0f}%)")

# ─────────────────────────────────────────────────────────────────────────────
# 5. ENTRENAR EL MODELO DE REGRESIÓN LINEAL
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 65)
print("   ENTRENAMIENTO DEL MODELO")
print("=" * 65)

modelo = LinearRegression()
modelo.fit(X_entrenamiento, y_entrenamiento)
print("\n[OK] Modelo entrenado exitosamente.")

# Coeficientes e intercepto
print("\nCOEFICIENTES DEL MODELO (interpretación):")
print(f"  Intercepto (valor base): {modelo.intercept_:.4f}")
for nombre, coef in zip(X.columns, modelo.coef_):
    print(f"  {nombre:20s}: {coef:+.4f}  -> por cada punto más en esta carta, la suma aumenta {abs(coef):.2f}")

# ─────────────────────────────────────────────────────────────────────────────
# 6. PREDICCIONES Y MÉTRICAS
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 65)
print("   EVALUACIÓN DEL MODELO")
print("=" * 65)

predicciones = modelo.predict(X_prueba)

mae  = mean_absolute_error(y_prueba, predicciones)
mse  = mean_squared_error(y_prueba, predicciones)
rmse = np.sqrt(mse)
r2   = r2_score(y_prueba, predicciones)

print(f"\nMÉTRICAS DE RENDIMIENTO:")
print(f"  MAE  (Error Absoluto Medio):       {mae:.4f}")
print(f"  MSE  (Error Cuadrático Medio):     {mse:.4f}")
print(f"  RMSE (Raíz del Error Cuadrático):  {rmse:.4f}")
print(f"  R²   (Coeficiente de Determinación): {r2:.4f}  ({r2*100:.2f}%)")
print(f"\n  Interpretación:")
print(f"  El modelo explica el {r2*100:.2f}% de la variabilidad de la carta máxima.")
print(f"  En promedio, el error de predicción es de {mae:.2f} puntos.")

# ─────────────────────────────────────────────────────────────────────────────
# 7. PREDICCIÓN DE UN CASO NUEVO
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 65)
print("   PREDICCIÓN — CASO NUEVO")
print("=" * 65)

# Mano de ejemplo: As (1), Rey (13), Reina (12), Jota (11), 10
nuevo_jugador = pd.DataFrame({
    "carta1": [1],
    "carta2": [13],
    "carta3": [12],
    "carta4": [11],
    "carta5": [10],
    "cartas_distintas": [5],
    "clase_mano": [0]
})

carta_max_real = 13
prediccion_nueva = modelo.predict(nuevo_jugador)[0]

print(f"\nMano del nuevo jugador: As(1), Rey(13), Reina(12), Jota(11), 10")
print(f"  Carta máxima real:      {carta_max_real}")
print(f"  Predicción del modelo:  {prediccion_nueva:.2f}")
print(f"  Diferencia:             {abs(carta_max_real - prediccion_nueva):.2f}")

# Varios casos nuevos en tabla
print("\nPREDICCIÓN DE VARIOS CASOS NUEVOS:")
casos = pd.DataFrame({
    "carta1":         [1,  5,  7,  13,  2],
    "carta2":         [13, 5,  7,  13,  2],
    "carta3":         [12, 5,  8,  13,  3],
    "carta4":         [11, 6,  9,  13,  4],
    "carta5":         [10, 6,  10, 12,  5],
    "cartas_distintas":[5, 3,  4,  2,   4],
    "clase_mano":     [0,  2,  1,  7,   1]
})
casos["carta_maxima_real"] = casos[["carta1","carta2","carta3","carta4","carta5"]].max(axis=1)
casos["prediccion"] = modelo.predict(casos[X.columns]).round(2)
casos["error"]      = (casos["prediccion"] - casos["carta_maxima_real"]).abs().round(2)
print(casos[["carta1","carta2","carta3","carta4","carta5","carta_maxima_real","prediccion","error"]].to_string(index=False))

# ─────────────────────────────────────────────────────────────────────────────
# 8. GRÁFICAS
# ─────────────────────────────────────────────────────────────────────────────

# Gráfica 1: Valores reales vs predichos
fig, ax = plt.subplots(figsize=(8, 6))
ax.scatter(y_prueba, predicciones, alpha=0.35, color="#4C72B0", s=12, label="Predicciones")
min_val = min(y_prueba.min(), predicciones.min())
max_val = max(y_prueba.max(), predicciones.max())
ax.plot([min_val, max_val], [min_val, max_val], "r--", linewidth=2, label="Línea ideal (y=x)")
ax.set_title("Regresión Lineal: Valores Reales vs Predichos\n(Carta Máxima)", fontsize=13, fontweight="bold")
ax.set_xlabel("Carta Máxima Real", fontsize=11)
ax.set_ylabel("Carta Máxima Predicha", fontsize=11)
ax.legend()
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(RUTA_FIGURAS, "04_regresion_real_vs_predicho.png"), dpi=150)
plt.show()
print("\n[OK] Gráfica guardada: 04_regresion_real_vs_predicho.png")

# Gráfica 2: Residuos
residuos = y_prueba.values - predicciones
fig, ax = plt.subplots(figsize=(8, 5))
ax.scatter(predicciones, residuos, alpha=0.35, color="#DD8452", s=12)
ax.axhline(0, color="red", linestyle="--", linewidth=1.8, label="Residuo = 0")
ax.set_title("Gráfica de Residuos — Regresión Lineal", fontsize=13, fontweight="bold")
ax.set_xlabel("Valores Predichos", fontsize=11)
ax.set_ylabel("Residuos (Real - Predicho)", fontsize=11)
ax.legend()
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(RUTA_FIGURAS, "05_regresion_residuos.png"), dpi=150)
plt.show()
print("[OK] Gráfica guardada: 05_regresion_residuos.png")

# Gráfica 3: Importancia de coeficientes (barras)
fig, ax = plt.subplots(figsize=(8, 5))
coefs = pd.Series(modelo.coef_, index=X.columns)
colores_barra = ["#2ca02c" if c > 0 else "#d62728" for c in coefs]
ax.barh(coefs.index, coefs.values, color=colores_barra, edgecolor="white")
ax.set_title("Coeficientes del Modelo de Regresión Lineal", fontsize=13, fontweight="bold")
ax.set_xlabel("Valor del Coeficiente", fontsize=11)
ax.axvline(0, color="black", linewidth=0.8)
ax.grid(axis="x", alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(RUTA_FIGURAS, "06_regresion_coeficientes.png"), dpi=150)
plt.show()
print("[OK] Gráfica guardada: 06_regresion_coeficientes.png")

# ─────────────────────────────────────────────────────────────────────────────
# 9. RESUMEN FINAL
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 65)
print("   RESUMEN DE REGRESIÓN LINEAL")
print("=" * 65)
print(f"  Variable objetivo:     carta_maxima")
print(f"  Variables usadas:      {len(X.columns)}")
print(f"  MAE:   {mae:.4f}")
print(f"  RMSE:  {rmse:.4f}")
print(f"  R²:    {r2:.4f}  ({r2*100:.2f}%)")
print("=" * 65)
print("   REGRESIÓN LINEAL COMPLETADA")
print("=" * 65)

