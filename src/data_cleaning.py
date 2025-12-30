import pandas as pd
import numpy as np
import re

# ==========================================
# 0. CARGA INICIAL
# ==========================================

ruta_fichero = "dataset/Global Data Analyst Job Market 2025.csv"
df = pd.read_csv(ruta_fichero, sep=";")

print(f"Dimensiones Originales: {df.shape}")

# ==========================================
# 3.1. GESTIÓN DE CEROS Y NULOS
# ==========================================

print("\n--- 3.1 Análisis de Nulos ---")
cols_criticas = ['salario_medio_ppp_2024', 'indice_coste_vida_2024']
df_clean = df.dropna(subset=cols_criticas).copy()

df_clean['titulo'] = df_clean['titulo'].fillna('Sin Título')
df_clean['empresa'] = df_clean['empresa'].fillna('Empresa Confidencial')
df_clean = df_clean[df_clean['indice_coste_vida_2024'] > 0]

# ==========================================
# 3.2. GESTIÓN DE TIPOS DE DATOS
# ==========================================

print("\n--- 3.2 Conversión de Tipos ---")
cols_categoricas = ['pais', 'modalidad']
for col in cols_categoricas:
    df_clean[col] = df_clean[col].astype('category')

df_clean['empresa'] = df_clean['empresa'].astype(str)
df_clean['es_teletrabajo'] = df_clean['es_teletrabajo'].astype(bool)

# ==========================================
# 3.3. GESTIÓN DE OUTLIERS 
# ==========================================

print("\n--- 3.3 Gestión de Outliers ---")

# --- PASO 1: FILTRO DE CALIDAD (MÍNIMO 20 CARACTERES) ---
# CAMBIO CLAVE: Bajamos a 20 .
# Solo borramos la basura real (0, 8 chars, etc.)
filtro_calidad = df_clean['desc_longitud'] >= 20
borrados = (~filtro_calidad).sum()
df_clean = df_clean[filtro_calidad]

print(f"   -> Se han eliminado {borrados} filas por descripción nula o error (<20 chars).")
print(f"   -> Mínimo actual: {df_clean['desc_longitud'].min()} (Debe ser >= 20)")

# --- PASO 2: CAPPING ESTADÍSTICO (IQR) ---
# Calculamos cuartiles sobre los datos limpios
Q1 = df_clean['desc_longitud'].quantile(0.25)
Q3 = df_clean['desc_longitud'].quantile(0.75)
IQR = Q3 - Q1
upper_bound = Q3 + 1.5 * IQR

# Aplicamos Capping solo superior
df_clean['desc_longitud'] = np.where(
    df_clean['desc_longitud'] > upper_bound, 
    upper_bound, 
    df_clean['desc_longitud']
)

print(f"   -> Winsorization Superior aplicada. Límite: {upper_bound:.2f}")

# ==========================================
# 3.4. LIMPIEZA DE TEXTO (CORREGIDA)
# ==========================================

print("\n--- 3.4 Normalización de Texto ---")

df_clean['titulo'] = df_clean['titulo'].str.strip().str.title()

def limpiar_ubicacion_regex_final(texto):
    if pd.isna(texto): return "Desconocido"
    texto = str(texto)
    
    if ',' in texto: texto = texto.split(',')[0]
    
    patron_ruido = r'(?i)\b(teletrabajo|trabajo|híbrido|hybrid|remote|homeoffice|work| in | en | at |España|Spain|Deutschland|Germany|United States|USA|UK|France|Francia)\b'
    texto_limpio = re.sub(patron_ruido, ' ', texto)
    
    texto_limpio = re.sub(r'\b\d{4,5}\b', '', texto_limpio)
    texto_limpio = re.sub(r'\s+', ' ', texto_limpio).strip().title()
    
    if len(texto_limpio) < 2: return "Desconocido"
    
    return texto_limpio

df_clean['ciudad_limpia'] = df_clean['ubicacion_raw'].apply(limpiar_ubicacion_regex_final)

# Recálculo variable objetivo
df_clean['salario_real_ajustado'] = (
    df_clean['salario_medio_ppp_2024'] / (df_clean['indice_coste_vida_2024'] / 100)
).round(2)

# ==========================================
# GUARDADO FINAL
# ==========================================

nombre_salida = "dataset/Global Data Analyst Job Market_Clean.csv"
df_clean.to_csv(nombre_salida, index=False, sep=";", encoding="utf-8-sig")

print(f"\n[OK] Dataset limpio guardado en: {nombre_salida}")
print(f"Filas finales: {len(df_clean)}")