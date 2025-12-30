from pathlib import Path

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from scipy import stats

# ==============================================================================
# 0. CARGA DE DATOS
# ==============================================================================

ruta_fichero = "dataset/Global Data Analyst Job Market_Clean.csv"
df = pd.read_csv(ruta_fichero, sep=";")
print(f"Datos cargados para análisis: {df.shape}")
fig_dir = Path("dataset") / "figs"
fig_dir.mkdir(parents=True, exist_ok=True)

# ==============================================================================
# 4.1.A. MODELO NO SUPERVISADO (CLUSTERING ECONÓMICO)
# Objetivo: Identificar perfiles de rentabilidad (Salario vs Coste)
# ==============================================================================

print("\n--- 4.1.A Clustering Económico (K-Means) ---")

# 1. Selección y Escalado
X_cluster = df[['salario_real_ajustado', 'indice_coste_vida_2024']].copy()
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_cluster)

# 2. Aplicación del Modelo
kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
df['cluster_label'] = kmeans.fit_predict(X_scaled)

# 3. Asignación Inteligente de Etiquetas (Lógica de Negocio)
# Calculamos medias para identificar qué cluster es cual
resumen = df.groupby('cluster_label')[['salario_real_ajustado', 'indice_coste_vida_2024']].mean()

# Lógica:
# - Alta Eficiencia: El que tiene menor coste de vida (idxmin) dentro de los competitivos
id_eficiente = resumen['indice_coste_vida_2024'].idxmin()
# - Coste Elevado: El que tiene el mayor coste de vida (idxmax)
id_caro = resumen['indice_coste_vida_2024'].idxmax()
# - Retorno Limitado: El que queda
id_resto = list(set(resumen.index) - {id_eficiente, id_caro})[0]

mapa_nombres = {
    id_eficiente: 'Alta Eficiencia (Coste Bajo/Salario Alto)', 
    id_caro:      'Altos Ingresos / Coste Elevado',
    id_resto:     'Retorno Limitado'
}
df['cluster_nombre'] = df['cluster_label'].map(mapa_nombres)

# 4. Visualización Definitiva
plt.figure(figsize=(11, 7))
sns.scatterplot(
    data=df, 
    x='indice_coste_vida_2024', 
    y='salario_real_ajustado', 
    hue='cluster_nombre',    
    style='pais',            
    palette='viridis', 
    s=140, 
    alpha=0.85
)

plt.title('Matriz de Rentabilidad Real: ¿Dónde compensa trabajar?', fontsize=14, fontweight='bold')
plt.xlabel('Índice Coste de Vida (Menor es mejor)', fontsize=12)
plt.ylabel('Salario Real Ajustado (Poder Adquisitivo)', fontsize=12)
plt.legend(bbox_to_anchor=(1.01, 1), loc='upper left', title="Perfil de Mercado")
plt.grid(True, linestyle='--', alpha=0.4)
plt.tight_layout()
plt.savefig(fig_dir / 'kmeans_clusters.png', dpi=300)
plt.close()

# Resumen numérico para la memoria
print("\n>> Perfil de los Clusters Identificados:")
print(df.groupby('cluster_nombre')[['salario_real_ajustado', 'indice_coste_vida_2024']].mean())


# ==============================================================================
# 4.1.B. MODELO SUPERVISADO (RANDOM FOREST)
# Objetivo: Predecir Teletrabajo priorizando la detección de oportunidades (Recall)
# ==============================================================================

print("\n--- 4.1.B Modelo Supervisado (Random Forest) ---")

# 1. Ingeniería de Variables
df['titulo_len'] = df['titulo'].str.len()

# 2. Preparación (X, y)
X = pd.get_dummies(df[['desc_longitud', 'titulo_len', 'pais']], drop_first=True)
y = df['es_teletrabajo']

# 3. Split Train/Test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. Entrenamiento (Balanced para detectar la clase minoritaria)
rf_model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
rf_model.fit(X_train, y_train)

# 5. Evaluación
y_pred = rf_model.predict(X_test)
acc = accuracy_score(y_test, y_pred)

print(f"\n>> EXACTITUD GLOBAL (Accuracy): {acc:.2%}")
print("\n>> REPORTE DE CLASIFICACIÓN (Atención al Recall de 'True'):")
print(classification_report(y_test, y_pred))

# Importancia de variables (Opcional, para justificar)
importances = pd.Series(rf_model.feature_importances_, index=X.columns).nlargest(3)
print("\n>> Variables más influyentes en la predicción:")
print(importances)


# ==============================================================================
# 4.2. CONTRASTE DE HIPÓTESIS
# Pregunta: ¿Son diferentes las descripciones de ofertas remotas vs presenciales?
# ==============================================================================

print("\n--- 4.2 Contraste de Hipótesis ---")

# 1. Definición de grupos
grupo_remoto = df[df['es_teletrabajo'] == True]['desc_longitud']
grupo_presencial = df[df['es_teletrabajo'] == False]['desc_longitud']

print(f"Longitud Media - Remoto:     {grupo_remoto.mean():.2f} caracteres")
print(f"Longitud Media - Presencial: {grupo_presencial.mean():.2f} caracteres")

# 2. Test de Normalidad (Shapiro-Wilk)
_, p_norm_r = stats.shapiro(grupo_remoto)
_, p_norm_p = stats.shapiro(grupo_presencial)

print(f"\nTest de Normalidad (Shapiro): p_remoto={p_norm_r:.5f}, p_presencial={p_norm_p:.5f}")

# 3. Selección y Ejecución del Test
alpha = 0.05
if p_norm_r < alpha or p_norm_p < alpha:
    print(">> Decisión: Datos NO normales -> Se aplica U de Mann-Whitney.")
    stat, p_val = stats.mannwhitneyu(grupo_remoto, grupo_presencial, alternative='two-sided')
else:
    print(">> Decisión: Datos Normales -> Se aplica T-Student.")
    stat, p_val = stats.ttest_ind(grupo_remoto, grupo_presencial)

print(f"\n>> RESULTADO DEL CONTRASTE: p-value = {p_val:.5f}")

if p_val < alpha:
    print(">>> CONCLUSIÓN: RECHAZAMOS H0. Existen diferencias significativas entre grupos.")
else:
    print(">>> CONCLUSIÓN: NO RECHAZAMOS H0. No hay evidencia suficiente de diferencias.")

# Gráfico Boxplot para la memoria
plt.figure(figsize=(8, 5))
sns.boxplot(
    data=df,
    x='es_teletrabajo',
    y='desc_longitud',
    hue='es_teletrabajo',
    palette='pastel',
    legend=False
)
plt.title('Distribución de Longitud de Descripción por Modalidad')
plt.xlabel('¿Es Teletrabajo?')
plt.ylabel('Caracteres (Longitud)')
plt.grid(True, axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig(fig_dir / 'boxplot_desc_vs_modalidad.png', dpi=300)
plt.close()
