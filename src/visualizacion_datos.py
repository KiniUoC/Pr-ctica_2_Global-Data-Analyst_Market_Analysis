import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import os

# Configuración Estética General
sns.set_theme(style="whitegrid")
plt.rcParams['figure.figsize'] = (10, 6)

# ==============================================================================
# 0. CONFIGURACIÓN Y CARGA
# ==============================================================================
# Crear carpeta para las figuras si no existe
output_folder = "figs"
os.makedirs(output_folder, exist_ok=True)
print(f">>> Carpeta de salida configurada: ./{output_folder}/")

print(">>> Cargando dataset y preparando variables...")
# Asegúrate de que la ruta al CSV sea correcta en tu ordenador
df = pd.read_csv("dataset/Global Data Analyst Job Market_Clean.csv", sep=";")

# Ingeniería de variables necesaria para los modelos
df['titulo_len'] = df['titulo'].str.len()

# ==============================================================================
# BLOQUE 1: ANÁLISIS DESCRIPTIVO (CONTEXTO)
# ==============================================================================

# --- FIGURA 1: Contexto de Mercado (Volumen y Salarios) ---
print(">>> Generando Figura 1 (Contexto)...")
plt.figure(figsize=(12, 6))

# A) Volumen de Ofertas
plt.subplot(1, 2, 1)
ax1 = sns.countplot(data=df, x='pais', palette='viridis', order=df['pais'].value_counts().index)
plt.title('A) Volumen de Ofertas por País', fontsize=12, fontweight='bold')
plt.xlabel('Mercado')
plt.ylabel('Nº Ofertas')
plt.bar_label(ax1.containers[0])

# B) Salario Real (BARRAS)
plt.subplot(1, 2, 2)
ax2 = sns.barplot(data=df, x='pais', y='salario_real_ajustado', palette='magma', errorbar=None)
plt.title('B) Salario Real Ajustado (Poder de Compra)', fontsize=12, fontweight='bold')
plt.ylabel('USD (PPP)')
plt.xlabel('Mercado')

# Etiquetas de valor encima de las barras
for container in ax2.containers:
    ax2.bar_label(container, fmt='%.0f $', padding=3, fontsize=10)

plt.tight_layout()
plt.savefig(f"{output_folder}/Figura_1_Contexto_Mercado.png", dpi=300)
plt.show()

# --- FIGURA 2: Proporción de Teletrabajo ---
print(">>> Generando Figura 2 (Teletrabajo)...")
plt.figure(figsize=(6, 6))
counts = df['es_teletrabajo'].value_counts()
plt.pie(counts, labels=['Presencial', 'Teletrabajo'], autopct='%1.1f%%', 
        colors=['#ff9999','#66b3ff'], startangle=90, explode=(0.05, 0))
plt.title('Proporción Global de Teletrabajo', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig(f"{output_folder}/Figura_2_Proporcion_Teletrabajo.png", dpi=300)
plt.show()

# ==============================================================================
# BLOQUE 2: VALIDACIÓN ESTADÍSTICA
# ==============================================================================

# --- FIGURA 3: Test Visual de Normalidad ---
print(">>> Generando Figura 3 (Normalidad)...")
plt.figure(figsize=(12, 5))

# A) Histograma + KDE
plt.subplot(1, 2, 1)
sns.histplot(df['desc_longitud'], kde=True, color='teal', bins=25, alpha=0.6)
plt.axvline(df['desc_longitud'].mean(), color='red', linestyle='--', label='Media')
plt.axvline(df['desc_longitud'].median(), color='green', linestyle='-', label='Mediana')
plt.title('Distribución Asimétrica (Skewness)', fontsize=12)
plt.xlabel('Longitud Descripción (Caracteres)')
plt.legend()

# B) Q-Q Plot
plt.subplot(1, 2, 2)
stats.probplot(df['desc_longitud'], dist="norm", plot=plt)
plt.title('Gráfico Q-Q (No Normalidad)', fontsize=12)
plt.xlabel('Cuantiles Teóricos')
plt.ylabel('Valores Observados')

plt.tight_layout()
plt.savefig(f"{output_folder}/Figura_3_Validacion_Normalidad.png", dpi=300)
plt.show()

# ==============================================================================
# BLOQUE 3: RESULTADOS DEL ANÁLISIS AVANZADO
# ==============================================================================

# --- Recálculo K-Means para graficar ---
X_cluster = df[['salario_real_ajustado', 'indice_coste_vida_2024']].copy()
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_cluster)
kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
df['cluster_label'] = kmeans.fit_predict(X_scaled)

# Asignación de Nombres (Lógica: Menor Coste = Mayor Eficiencia)
resumen = df.groupby('cluster_label')[['salario_real_ajustado', 'indice_coste_vida_2024']].mean()
id_eficiente = resumen['indice_coste_vida_2024'].idxmin() # España
id_caro = resumen['indice_coste_vida_2024'].idxmax()      # USA
id_resto = list(set(resumen.index) - {id_eficiente, id_caro})[0]

mapa = {id_eficiente: 'Alta Eficiencia', id_caro: 'Coste Elevado', id_resto: 'Retorno Limitado'}
df['cluster_nombre'] = df['cluster_label'].map(mapa)

# --- FIGURA 4: Mapa de Clusters ---
print(">>> Generando Figura 4 (Clusters)...")
plt.figure(figsize=(10, 7))
sns.scatterplot(data=df, x='indice_coste_vida_2024', y='salario_real_ajustado', 
                hue='cluster_nombre', style='pais', palette='viridis', s=150, alpha=0.9)
plt.title('Mapa de Rentabilidad: Identificación de Clusters', fontsize=13, fontweight='bold')
plt.xlabel('Índice Coste de Vida (Menor es mejor)')
plt.ylabel('Salario Real Ajustado (Mayor es mejor)')
plt.legend(bbox_to_anchor=(1.01, 1), loc='upper left', title="Perfil Económico")
plt.tight_layout()
plt.savefig(f"{output_folder}/Figura_4_Clusters_Economicos.png", dpi=300)
plt.show()

# --- Recálculo Random Forest para graficar ---
X = pd.get_dummies(df[['desc_longitud', 'titulo_len', 'pais']], drop_first=True)
y = df['es_teletrabajo']
rf = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
rf.fit(X, y)

# --- FIGURA 5: Feature Importance ---
print(">>> Generando Figura 5 (Feature Importance)...")
importances = pd.Series(rf.feature_importances_, index=X.columns).nlargest(5).sort_values()
plt.figure(figsize=(8, 5))
importances.plot(kind='barh', color='#86bf91')
plt.title('Variables Predictoras del Teletrabajo', fontsize=13, fontweight='bold')
plt.xlabel('Peso en el Modelo (Importancia Relativa)')
plt.tight_layout()
plt.savefig(f"{output_folder}/Figura_5_Feature_Importance.png", dpi=300)
plt.show()

print(f"\n[ÉXITO] Las 5 figuras se han guardado en la carpeta '{output_folder}'.")