






import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import os




RUTA_FIGURAS = os.path.join("..", "reports", "figures")
RUTA_CLEAN   = os.path.join("..", "data", "clean")
os.makedirs(RUTA_FIGURAS, exist_ok=True)




print("=" * 65)
print("   ÁRBOL DE DECISIÓN — Predicción: ¿Ganó o Perdió?")
print("=" * 65)

ruta_csv = os.path.join(RUTA_CLEAN, "poker_manos_clean.csv")
df = pd.read_csv(ruta_csv)

print(f"\nDataset cargado: {df.shape[0]:,} filas x {df.shape[1]} columnas")
print(f"\nDistribución de la variable objetivo (gano):")
print(df["gano"].value_counts().rename({0: "Perdio (0)", 1: "Gano (1)"}).to_string())
print(f"  % ganadoras: {df['gano'].mean()*100:.2f}%")




print("\n" + "-" * 50)
print("VARIABLES DEL MODELO:")
print("-" * 50)

X = df[["carta1", "carta2", "carta3", "carta4", "carta5",
        "carta_maxima", "cartas_distintas", "suma_cartas"]]
y = df["gano"]

print("Variables predictoras (X):")
for col in X.columns:
    print(f"  - {col}")
print(f"\nVariable objetivo (y): gano  (0=perdio, 1=gano)")




X_entrenamiento, X_prueba, y_entrenamiento, y_prueba = train_test_split(
    X, y, test_size=0.3, random_state=42
)

print(f"\nDIVISIÓN DEL DATASET:")
print(f"  Entrenamiento: {X_entrenamiento.shape[0]:,} filas (70%)")
print(f"  Prueba:        {X_prueba.shape[0]:,} filas (30%)")




print("\n" + "=" * 65)
print("   ENTRENAMIENTO DEL ÁRBOL DE DECISIÓN")
print("=" * 65)





modelo = DecisionTreeClassifier(max_depth=5, random_state=42)
modelo.fit(X_entrenamiento, y_entrenamiento)

print("\n[OK] Árbol de decisión entrenado.")
print(f"  max_depth = 5  (limitado para evitar sobreajuste)")
print(f"  Profundidad real del árbol: {modelo.get_depth()}")
print(f"  Número de hojas (nodos terminales): {modelo.get_n_leaves()}")




print("\n" + "=" * 65)
print("   EVALUACIÓN DEL MODELO")
print("=" * 65)

predicciones = modelo.predict(X_prueba)
exactitud = accuracy_score(y_prueba, predicciones)

print(f"\nEXACTITUD DEL MODELO: {exactitud*100:.2f}%")
print(f"  El árbol clasifica correctamente el {exactitud*100:.2f}% de las manos.")

print("\nMATRIZ DE CONFUSIÓN:")
mc = confusion_matrix(y_prueba, predicciones)
mc_df = pd.DataFrame(mc,
    index=["Real: Perdio", "Real: Gano"],
    columns=["Pred: Perdio", "Pred: Gano"]
)
print(mc_df.to_string())

print(f"\n  Verdaderos Negativos (TN): {mc[0,0]}  (predijo perder, y perdio)")
print(f"  Falsos Positivos   (FP): {mc[0,1]}  (predijo ganar, pero perdio)")
print(f"  Falsos Negativos   (FN): {mc[1,0]}  (predijo perder, pero gano)")
print(f"  Verdaderos Positivos(TP): {mc[1,1]}  (predijo ganar, y gano)")

print("\nREPORTE DE CLASIFICACIÓN:")
print(classification_report(y_prueba, predicciones,
      target_names=["Perdio (0)", "Gano (1)"]))




print("\nIMPORTANCIA DE VARIABLES:")
importancias = pd.Series(modelo.feature_importances_, index=X.columns)
importancias = importancias.sort_values(ascending=False)
for var, imp in importancias.items():
    print(f"  {var:22s}: {imp:.4f}  ({imp*100:.2f}%)")




print("\n" + "=" * 65)
print("   PREDICCIÓN — CASOS NUEVOS")
print("=" * 65)


nuevo_jugador = pd.DataFrame({
    "carta1": [1], "carta2": [13], "carta3": [12],
    "carta4": [11], "carta5": [10],
    "carta_maxima": [13], "cartas_distintas": [5], "suma_cartas": [47]
})

pred_clase = modelo.predict(nuevo_jugador)[0]
prob = modelo.predict_proba(nuevo_jugador)[0]
resultado = "GANARA (tiene mano buena)" if pred_clase == 1 else "NO GANARA (carta alta)"
print(f"\nMano: As(1), Rey(13), Reina(12), Jota(11), 10")
print(f"  Prediccion:   {resultado}")
print(f"  Probabilidad de ganar:  {prob[1]*100:.1f}%")
print(f"  Probabilidad de perder: {prob[0]*100:.1f}%")


nuevos_casos = pd.DataFrame({
    "carta1":         [1,  5,  7,  13,  2,  3],
    "carta2":         [13, 5,  7,  13,  2,  3],
    "carta3":         [12, 5,  8,  13,  3,  3],
    "carta4":         [11, 6,  9,  13,  4,  4],
    "carta5":         [10, 6,  10, 12,  5,  4],
    "carta_maxima":   [13, 6,  10, 13,  5,  4],
    "cartas_distintas":[5, 3,  4,  2,   4,  2],
    "suma_cartas":    [47, 27, 41, 64,  16, 17]
})
nuevos_casos["prediccion"] = modelo.predict(nuevos_casos[X.columns])
nuevos_casos["resultado"]  = nuevos_casos["prediccion"].map({1: "GANARA", 0: "NO GANARA"})
probs = modelo.predict_proba(nuevos_casos[X.columns])
nuevos_casos["prob_ganar"] = (probs[:, 1] * 100).round(1)

print("\nPREDICCIÓN DE VARIOS CASOS NUEVOS:")
print(nuevos_casos[["carta1","carta2","carta3","carta4","carta5","suma_cartas","resultado","prob_ganar"]].to_string(index=False))






fig, ax = plt.subplots(figsize=(20, 8))
plot_tree(modelo, feature_names=X.columns.tolist(),
          class_names=["Perdio", "Gano"],
          filled=True, rounded=True, fontsize=9, ax=ax,
          impurity=False, proportion=True)
ax.set_title("Árbol de Decisión — Predicción de Mano Ganadora en Póker",
             fontsize=14, fontweight="bold")
plt.tight_layout()
plt.savefig(os.path.join(RUTA_FIGURAS, "07_arbol_decision.png"), dpi=120, bbox_inches="tight")
print("\n[OK] Gráfica guardada: 07_arbol_decision.png")


fig, ax = plt.subplots(figsize=(8, 5))
colores = plt.cm.RdYlGn(np.linspace(0.3, 0.9, len(importancias)))
ax.barh(importancias.index, importancias.values, color=colores[::-1])
ax.set_title("Importancia de Variables — Árbol de Decisión", fontsize=13, fontweight="bold")
ax.set_xlabel("Importancia (Gini)", fontsize=11)
ax.grid(axis="x", alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(RUTA_FIGURAS, "08_importancia_variables.png"), dpi=150)
print("[OK] Gráfica guardada: 08_importancia_variables.png")


fig, ax = plt.subplots(figsize=(10, 4))
colores_pred = ["#2ca02c" if r == "GANARA" else "#d62728" for r in nuevos_casos["resultado"]]
etiquetas = [f"Mano {i+1}\nSuma={s}" for i, s in enumerate(nuevos_casos["suma_cartas"])]
barras = ax.bar(etiquetas, nuevos_casos["prob_ganar"], color=colores_pred, edgecolor="white", linewidth=1.2)
ax.set_title("Probabilidad de Ganar — Nuevos Jugadores", fontsize=13, fontweight="bold")
ax.set_ylabel("Probabilidad de Ganar (%)", fontsize=11)
ax.set_ylim(0, 110)
ax.axhline(50, color="gray", linestyle="--", linewidth=1, alpha=0.7, label="Umbral 50%")
for barra, prob in zip(barras, nuevos_casos["prob_ganar"]):
    ax.text(barra.get_x() + barra.get_width()/2, barra.get_height() + 2,
            f"{prob:.1f}%", ha="center", va="bottom", fontsize=10, fontweight="bold")
ax.legend()
ax.grid(axis="y", alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(RUTA_FIGURAS, "09_predicciones_nuevos_jugadores.png"), dpi=150)
print("[OK] Gráfica guardada: 09_predicciones_nuevos_jugadores.png")


fig, ax = plt.subplots(figsize=(5, 4))
im = ax.imshow(mc, interpolation="nearest", cmap="Blues")
ax.set_title("Matriz de Confusión", fontsize=13, fontweight="bold")
ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
ax.set_xticklabels(["Pred: Perdio", "Pred: Gano"]); ax.set_yticklabels(["Real: Perdio", "Real: Gano"])
for i in range(2):
    for j in range(2):
        ax.text(j, i, str(mc[i, j]), ha="center", va="center", fontsize=16,
                color="white" if mc[i, j] > mc.max()/2 else "black")
plt.tight_layout()
plt.savefig(os.path.join(RUTA_FIGURAS, "10_matriz_confusion.png"), dpi=150)
print("[OK] Gráfica guardada: 10_matriz_confusion.png")




print("\n" + "=" * 65)
print("   RESUMEN DE ÁRBOL DE DECISIÓN")
print("=" * 65)
print(f"  Variable objetivo:     gano (0=perdio, 1=gano)")
print(f"  max_depth:             5")
print(f"  Exactitud:             {exactitud*100:.2f}%")
print(f"  Variable más importante: {importancias.idxmax()} ({importancias.max()*100:.2f}%)")
print("=" * 65)
print("   ÁRBOL DE DECISIÓN COMPLETADO")
print("=" * 65)
