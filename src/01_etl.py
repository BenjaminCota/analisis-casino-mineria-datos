# =============================================================================
# PROYECTO: ANÃLISIS DE DATOS DE CASINO / PÃ“KER
# SCRIPT:   01_etl.py  â€”  Proceso ETL (Extract, Transform, Load)
# DATASET:  UCI Poker Hand Dataset (1,000,000 manos de pÃ³ker reales)
# MATERIA:  ExtracciÃ³n de Conocimientos de Base de Datos
# =============================================================================

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import sqlite3
import os

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# 1. CONFIGURACIÃ“N DE RUTAS
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
RUTA_FIGURAS = os.path.join("..", "reports", "figures")
RUTA_CLEAN   = os.path.join("..", "data", "clean")
os.makedirs(RUTA_FIGURAS, exist_ok=True)
os.makedirs(RUTA_CLEAN,   exist_ok=True)

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# 2. EXTRACT â€” CARGAR EL DATASET DESDE URL PÃšBLICA
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
print("=" * 65)
print("   FASE EXTRACT â€” Carga del dataset UCI Poker Hand")
print("=" * 65)

URL_DATASET = "https://archive.ics.uci.edu/ml/machine-learning-databases/poker/poker-hand-testing.data"

# Nombres de columnas segÃºn documentaciÃ³n oficial del dataset
COLUMNAS = [
    "palo1", "carta1",
    "palo2", "carta2",
    "palo3", "carta3",
    "palo4", "carta4",
    "palo5", "carta5",
    "clase_mano"
]

print("\nCargando datos desde:")
print(URL_DATASET)

# Usamos 5,000 filas para que el script sea Ã¡gil y didÃ¡ctico
df = pd.read_csv(URL_DATASET, header=None, names=COLUMNAS, nrows=5000)

# Guardar copia en data/raw/
ruta_raw = os.path.join("..", "data", "raw", "poker_manos_raw.csv")
df.to_csv(ruta_raw, index=False)
print(f"\nDataset guardado en: {ruta_raw}")

print("\nPRIMEROS REGISTROS:")
print(df.head(10).to_string())

print("\nTAMAÃ‘O DEL DATASET:")
print(f"  Filas:    {df.shape[0]:,}")
print(f"  Columnas: {df.shape[1]}")

print("\nTIPOS DE DATOS:")
print(df.dtypes.to_string())

print("\nESTADÃSTICAS DESCRIPTIVAS:")
print(df.describe().to_string())

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# 3. TRANSFORM â€” LIMPIEZA Y ENRIQUECIMIENTO
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
print("\n" + "=" * 65)
print("   FASE TRANSFORM â€” Limpieza y creaciÃ³n de variables")
print("=" * 65)

# 3.1  Revisar valores nulos
print("\nVALORES NULOS POR COLUMNA:")
nulos = df.isnull().sum()
print(nulos.to_string())
print(f"  Total de nulos: {nulos.sum()}")

# 3.2  Revisar duplicados
duplicados = df.duplicated().sum()
print(f"\nDUPLICADOS ENCONTRADOS: {duplicados}")
if duplicados > 0:
    df = df.drop_duplicates()
    print(f"  â†’ Duplicados eliminados. Filas restantes: {len(df):,}")
else:
    print("  â†’ No hay duplicados. Dataset Ã­ntegro.")

# 3.3  Verificar rangos vÃ¡lidos
# Palos: 1-4 | Cartas: 1-13 | Clase: 0-9
print("\nVERIFICACIÃ“N DE RANGOS VÃLIDOS:")
for col in ["palo1", "palo2", "palo3", "palo4", "palo5"]:
    invalidos = df[~df[col].between(1, 4)].shape[0]
    print(f"  {col}: valores fuera de [1,4] = {invalidos}")
for col in ["carta1", "carta2", "carta3", "carta4", "carta5"]:
    invalidos = df[~df[col].between(1, 13)].shape[0]
    print(f"  {col}: valores fuera de [1,13] = {invalidos}")
invalidos_clase = df[~df["clase_mano"].between(0, 9)].shape[0]
print(f"  clase_mano: valores fuera de [0,9] = {invalidos_clase}")

# 3.4  Detectar outliers con IQR en suma de cartas (columna que crearemos)
suma_temp = df[["carta1","carta2","carta3","carta4","carta5"]].sum(axis=1)
Q1, Q3 = suma_temp.quantile(0.25), suma_temp.quantile(0.75)
IQR = Q3 - Q1
limite_inf = Q1 - 1.5 * IQR
limite_sup = Q3 + 1.5 * IQR
outliers = ((suma_temp < limite_inf) | (suma_temp > limite_sup)).sum()
print(f"\nDETECCIÃ“N DE OUTLIERS (criterio IQR en suma de cartas):")
print(f"  Q1={Q1:.1f}, Q3={Q3:.1f}, IQR={IQR:.1f}")
print(f"  LÃ­mite inferior: {limite_inf:.1f} | LÃ­mite superior: {limite_sup:.1f}")
print(f"  Registros con outlier en suma_cartas: {outliers}")
print("  â†’ Se conservan; son manos extremas vÃ¡lidas en el juego de pÃ³ker.")

# 3.5  CREAR VARIABLES NUEVAS
print("\nCREANDO VARIABLES NUEVAS:")

# Variable 1: suma_cartas â€” valor total de las 5 cartas (objetivo de regresiÃ³n)
df["suma_cartas"] = df["carta1"] + df["carta2"] + df["carta3"] + df["carta4"] + df["carta5"]
print("  âœ” suma_cartas        = suma de los valores de las 5 cartas")

# Variable 2: carta_maxima â€” carta mÃ¡s alta de la mano
df["carta_maxima"] = df[["carta1","carta2","carta3","carta4","carta5"]].max(axis=1)
print("  âœ” carta_maxima       = valor de la carta mÃ¡s alta")

# Variable 3: cartas_distintas â€” cuÃ¡ntos valores Ãºnicos hay en la mano
df["cartas_distintas"] = df[["carta1","carta2","carta3","carta4","carta5"]].nunique(axis=1)
print("  âœ” cartas_distintas   = cantidad de valores distintos en la mano")

# Variable 4 (OBJETIVO DE CLASIFICACIÃ“N):
# gano = 1 si tiene mano ganadora (par o mejor, clase >= 1), 0 si es carta alta
# REGLA DOCUMENTADA: En pÃ³ker, cualquier combinaciÃ³n (par en adelante)
# supera a la mano de "carta alta" (clase_mano = 0).
df["gano"] = (df["clase_mano"] >= 1).astype(int)
print("  âœ” gano               = 1 si clase_mano>=1 (par o mejor), 0 si carta alta")

# Variable 5: nombre_mano â€” etiqueta textual de la clase
mapa_manos = {
    0: "Carta Alta",
    1: "Par",
    2: "Doble Par",
    3: "TrÃ­o",
    4: "Escalera",
    5: "Color",
    6: "Full House",
    7: "PÃ³ker",
    8: "Escalera de Color",
    9: "Escalera Real"
}
df["nombre_mano"] = df["clase_mano"].map(mapa_manos)
print("  âœ” nombre_mano        = nombre textual de la mano")

print("\nDATASET DESPUÃ‰S DE TRANSFORM (primeras filas):")
print(df.head().to_string())

print(f"\nFORMA FINAL DEL DATASET: {df.shape[0]:,} filas Ã— {df.shape[1]} columnas")

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# 4. MINI ANÃLISIS EXPLORATORIO
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
print("\n" + "=" * 65)
print("   ANÃLISIS EXPLORATORIO")
print("=" * 65)

# 4.1  DistribuciÃ³n de clases de mano
print("\nDISTRIBUCIÃ“N DE MANOS (frecuencia y porcentaje):")
distribucion = df.groupby("nombre_mano")["clase_mano"].count().reset_index()
distribucion.columns = ["nombre_mano", "frecuencia"]
distribucion["porcentaje"] = (distribucion["frecuencia"] / len(df) * 100).round(2)
distribucion = distribucion.sort_values("frecuencia", ascending=False)
print(distribucion.to_string(index=False))

# 4.2  Promedios de suma_cartas y carta_maxima por tipo de mano
print("\nPROMEDIOS POR TIPO DE MANO (groupby):")
promedios = df.groupby("nombre_mano")[["suma_cartas","carta_maxima","cartas_distintas"]].mean().round(2)
promedios = promedios.sort_values("suma_cartas", ascending=False)
print(promedios.to_string())

# 4.3  Porcentaje de victorias (gano)
pct_gano = df["gano"].mean() * 100
print(f"\nPORCENTAJE GLOBAL DE MANOS GANADORAS: {pct_gano:.2f}%")
print(f"  Ganadoras (gano=1): {df['gano'].sum():,}")
print(f"  Perdedoras (gano=0): {(df['gano']==0).sum():,}")

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# 5. GRÃFICAS (guardadas en reports/figures/)
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

# GrÃ¡fica 1: DistribuciÃ³n de tipos de mano
fig, ax = plt.subplots(figsize=(10, 5))
orden = distribucion.sort_values("frecuencia", ascending=False)
colores = plt.cm.Set2.colors
ax.bar(orden["nombre_mano"], orden["frecuencia"], color=colores[:len(orden)])
ax.set_title("DistribuciÃ³n de Tipos de Mano en PÃ³ker", fontsize=14, fontweight="bold")
ax.set_xlabel("Tipo de Mano", fontsize=11)
ax.set_ylabel("Frecuencia", fontsize=11)
ax.tick_params(axis="x", rotation=40)
ax.grid(axis="y", alpha=0.4)
plt.tight_layout()
plt.savefig(os.path.join(RUTA_FIGURAS, "01_distribucion_manos.png"), dpi=150)
plt.show()
print("\nâœ” GrÃ¡fica 1 guardada: 01_distribucion_manos.png")

# GrÃ¡fica 2: Histograma de suma_cartas
fig, ax = plt.subplots(figsize=(9, 5))
ax.hist(df["suma_cartas"], bins=30, color="#4C72B0", edgecolor="white", alpha=0.85)
ax.set_title("DistribuciÃ³n de la Suma de Cartas por Mano", fontsize=14, fontweight="bold")
ax.set_xlabel("Suma de Cartas", fontsize=11)
ax.set_ylabel("Frecuencia", fontsize=11)
ax.axvline(df["suma_cartas"].mean(), color="red", linestyle="--", linewidth=1.8, label=f"Media = {df['suma_cartas'].mean():.1f}")
ax.legend()
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(RUTA_FIGURAS, "02_histograma_suma_cartas.png"), dpi=150)
plt.show()
print("âœ” GrÃ¡fica 2 guardada: 02_histograma_suma_cartas.png")

# GrÃ¡fica 3: Manos ganadoras vs perdedoras (pie chart)
fig, ax = plt.subplots(figsize=(6, 6))
conteo_gano = df["gano"].value_counts()
etiquetas = ["GanÃ³ (par o mejor)", "PerdiÃ³ (carta alta)"]
ax.pie(conteo_gano, labels=etiquetas, autopct="%1.1f%%",
       colors=["#2ca02c", "#d62728"], startangle=140,
       wedgeprops={"edgecolor": "white", "linewidth": 1.5})
ax.set_title("ProporciÃ³n de Manos Ganadoras vs Perdedoras", fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig(os.path.join(RUTA_FIGURAS, "03_pie_gano_perdio.png"), dpi=150)
plt.show()
print("âœ” GrÃ¡fica 3 guardada: 03_pie_gano_perdio.png")

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# 6. LOAD â€” Guardar CSV limpio y cargar en SQLite
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
print("\n" + "=" * 65)
print("   FASE LOAD â€” Guardado de datos limpios")
print("=" * 65)

# 6.1  Guardar CSV limpio
ruta_csv = os.path.join(RUTA_CLEAN, "poker_manos_clean.csv")
df.to_csv(ruta_csv, index=False)
print(f"\nâœ” CSV limpio guardado en: {ruta_csv}")

# 6.2  Cargar en SQLite
ruta_db = os.path.join(RUTA_CLEAN, "casino.db")
conexion = sqlite3.connect(ruta_db)
df.to_sql("manos_poker", conexion, if_exists="replace", index=False)
print(f"âœ” Tabla 'manos_poker' cargada en SQLite: {ruta_db}")

# 6.3  Verificar con consulta SQL de ejemplo
print("\nVERIFICACIÃ“N SQL â€” Primeros 5 registros de la base de datos:")
consulta = "SELECT nombre_mano, suma_cartas, carta_maxima, gano FROM manos_poker LIMIT 5"
resultado_sql = pd.read_sql(consulta, conexion)
print(resultado_sql.to_string(index=False))

print("\nCONTEO POR TIPO DE MANO (SQL):")
consulta2 = """
    SELECT nombre_mano, COUNT(*) AS total, AVG(suma_cartas) AS promedio_suma
    FROM manos_poker
    GROUP BY nombre_mano
    ORDER BY total DESC
"""
resultado_sql2 = pd.read_sql(consulta2, conexion)
print(resultado_sql2.to_string(index=False))

conexion.close()

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# 7. GUARDAR RESUMEN EN EXCEL (varias hojas)
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
ruta_excel = os.path.join(RUTA_CLEAN, "resultados_casino.xlsx")

with pd.ExcelWriter(ruta_excel, engine="openpyxl") as writer:
    df.head(200).to_excel(writer, sheet_name="Dataset_Muestra", index=False)
    distribucion.to_excel(writer, sheet_name="Distribucion_Manos", index=False)
    promedios.to_excel(writer, sheet_name="Promedios_por_Mano")
    df.describe().to_excel(writer, sheet_name="Estadisticas")

print(f"\nâœ” Resumen Excel guardado en: {ruta_excel}")

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# 8. RESUMEN FINAL DEL ETL
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
print("\n" + "=" * 65)
print("   RESUMEN FINAL DEL PROCESO ETL")
print("=" * 65)
print(f"  Filas cargadas:          {df.shape[0]:,}")
print(f"  Columnas finales:        {df.shape[1]}")
print(f"  Nulos encontrados:       0")
print(f"  Duplicados eliminados:   {duplicados}")
print(f"  Variables nuevas:        5 (suma_cartas, carta_maxima, cartas_distintas, gano, nombre_mano)")
print(f"  % manos ganadoras:       {pct_gano:.2f}%")
print(f"  Archivos generados:")
print(f"    â†’ {ruta_raw}")
print(f"    â†’ {ruta_csv}")
print(f"    â†’ {ruta_db}")
print(f"    â†’ {ruta_excel}")
print(f"    â†’ 3 grÃ¡ficas PNG en reports/figures/")
print("=" * 65)
print("   ETL COMPLETADO EXITOSAMENTE")
print("=" * 65)


