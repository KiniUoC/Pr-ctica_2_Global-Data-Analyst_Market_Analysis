import pandas as pd
import os
import re
import io

# ==========================================
# 1. CONFIGURACIÓN DE RUTAS
# ==========================================
ruta_actual = os.path.dirname(os.path.abspath(__file__))
# Ajusta '..' si tu script está dentro de una subcarpeta src/
ruta_proyecto = os.path.dirname(ruta_actual) 
ruta_dataset = os.path.join(ruta_proyecto, "dataset")

# Fallback de seguridad
if not os.path.isdir(ruta_dataset):
    ruta_dataset = "dataset" if os.path.isdir("dataset") else "."

print(f"[INFO] Trabajando en: {ruta_dataset}")

# ==========================================
# 2. CARGA DE OFERTAS (INDEED)
# ==========================================
print("[INFO] Cargando Indeed...")
ruta_indeed = os.path.join(ruta_dataset, "indeed_global_final.csv")

try:
    df_indeed = pd.read_csv(
        ruta_indeed, 
        sep=',', 
        quotechar='"', 
        engine='python', 
        on_bad_lines='warn'
    )
    
    # 1. Selección de columnas de interés (INCLUYENDO desc_longitud y ubicacion_raw)
    cols_indeed_deseadas = [
        'titulo', 'empresa', 'pais', 'ubicacion_raw', 
        'modalidad', 'desc_longitud', 'url'
    ]
    
    # Filtrar solo las columnas que realmente existen en el CSV
    cols_existentes = [c for c in cols_indeed_deseadas if c in df_indeed.columns]
    df_indeed = df_indeed[cols_existentes]
    
    # Limpieza básica de espacios en país
    if 'pais' in df_indeed.columns:
        df_indeed['pais'] = df_indeed['pais'].str.strip()
        
    print(f"[OK] Indeed cargado: {len(df_indeed)} filas.")

except Exception as e:
    print(f"[ERROR] Fallo crítico cargando Indeed: {e}")
    exit()

# ==========================================
# 3. CARGA Y LIMPIEZA DE OECD (MACRO)
# ==========================================
print("[INFO] Procesando OECD...")

try:
    # Buscar archivo OECD automáticamente
    archivo_oecd = [f for f in os.listdir(ruta_dataset) if "OECD" in f and f.endswith(".csv")][0]
    df_oecd = pd.read_csv(os.path.join(ruta_dataset, archivo_oecd))
    
    # Mapa de códigos de país
    mapa_paises = {'ESP': 'ES', 'DEU': 'DE', 'GBR': 'UK', 'USA': 'US', 'FRA': 'FR'}
    
    # Detectar nombre de columna de país (suele variar entre REF_AREA y LOCATION)
    col_pais_oecd = 'REF_AREA' if 'REF_AREA' in df_oecd.columns else 'LOCATION'
    
    if col_pais_oecd in df_oecd.columns:
        df_oecd['pais'] = df_oecd[col_pais_oecd].map(mapa_paises)
        df_oecd = df_oecd.dropna(subset=['pais']) # Eliminar países fuera del alcance
        
        # Detectar columna de valor numérico
        col_valor = 'OBS_VALUE' if 'OBS_VALUE' in df_oecd.columns else 'Value'
        
        # Renombrar a algo legible
        df_oecd = df_oecd.rename(columns={col_valor: 'salario_medio_ppp_2024'})
        
        # --- LIMPIEZA VERTICAL ---
        # Nos quedamos SOLO con país y salario. Descartamos metadatos basura.
        df_oecd_limpio = df_oecd[['pais', 'salario_medio_ppp_2024']].copy()
        
        # Agrupar por país para evitar duplicados (media de valores si hay varios años)
        df_oecd_limpio = df_oecd_limpio.groupby('pais', as_index=False)['salario_medio_ppp_2024'].mean()
        
        print(f"[OK] OECD procesado. Países: {df_oecd_limpio['pais'].unique()}")
    else:
        print("[WARN] Columna de país no encontrada en OECD. Creando DF vacío.")
        df_oecd_limpio = pd.DataFrame(columns=['pais', 'salario_medio_ppp_2024'])

except Exception as e:
    print(f"[WARN] Error procesando OECD ({e}). Se omitirán datos salariales.")
    df_oecd_limpio = pd.DataFrame(columns=['pais', 'salario_medio_ppp_2024'])

# ==========================================
# 4. CARGA DE NUMBEO (Manual o CSV)
# ==========================================
ruta_numbeo = os.path.join(ruta_dataset, "datos_numbeo_manual.csv")
if os.path.exists(ruta_numbeo):
    with open(ruta_numbeo, "r", encoding="utf-8") as f:
        contenido = f.read().replace('"', '') # Limpieza rápida de comillas extra
    df_numbeo = pd.read_csv(io.StringIO(contenido), sep=";")
else:
    # Datos de respaldo (Fallback)
    df_numbeo = pd.DataFrame({
        'pais': ['ES', 'DE', 'UK', 'US', 'FR'],
        'indice_coste_vida_2024': [48.7, 63.5, 61.3, 72.9, 68.7],
        'indice_alquiler_2024': [18.2, 22.8, 26.9, 43.1, 21.5]
    })

# ==========================================
# 5. FUSIÓN DE DATOS (MERGE)
# ==========================================
print("[INFO] Fusionando datasets...")

# Merge 1: Fuentes Económicas (OECD + Numbeo)
df_macro = pd.merge(df_oecd_limpio, df_numbeo, on='pais', how='outer')

# Merge 2: Ofertas + Economía (Left Join para mantener todas las ofertas)
df_final = pd.merge(df_indeed, df_macro, on='pais', how='left')

# ==========================================
# 6. POST-PROCESADO Y LIMPIEZA FINAL
# ==========================================
# Rellenar nulos críticos
df_final['titulo'] = df_final['titulo'].fillna('Sin Título')

# Validar que exista la columna desc_longitud antes de convertirla
if 'desc_longitud' in df_final.columns:
    df_final['desc_longitud'] = pd.to_numeric(df_final['desc_longitud'], errors='coerce').fillna(0)

# Funciones de limpieza de texto
def limpiar_ciudad(texto):
    if pd.isna(texto): return "Desconocido"
    # Eliminar palabras clave comunes del scraping
    texto = re.sub(r'(?i)(teletrabajo|trabajo híbrido|homeoffice|remote|hybrid| in |au |en )', '', str(texto))
    texto = re.sub(r'\b\d{4,5}\b', '', texto) # Quitar códigos postales
    if "," in texto: texto = texto.split(",")[0] # Quedarse solo con la ciudad antes de la coma
    return texto.strip().title()

def detectar_remoto(texto):
    if pd.isna(texto): return False
    palabras = ['teletrabajo', 'remote', 'homeoffice', 'híbrido', 'hybrid']
    return any(p in str(texto).lower() for p in palabras)

if 'ubicacion_raw' in df_final.columns:
    df_final['es_teletrabajo'] = df_final['ubicacion_raw'].apply(detectar_remoto)
    df_final['ciudad_limpia'] = df_final['ubicacion_raw'].apply(limpiar_ciudad)
    # Estandarizar modalidad si se detecta remoto en el texto
    df_final.loc[df_final['es_teletrabajo'], 'modalidad'] = 'Remoto/Híbrido'

# ==========================================
# 6.5 CÁLCULO DE VARIABLE OBJETIVO (BIENESTAR)
# ==========================================
# Fórmula: Cuánto cunde el salario considerando el coste de vida local.
# Si el índice es 100 (NY), el salario vale lo que es. Si es 50 (barato), el salario cunde el doble.
df_final['salario_real_ajustado'] = df_final['salario_medio_ppp_2024'] / (df_final['indice_coste_vida_2024'] / 100)
df_final['salario_real_ajustado'] = df_final['salario_real_ajustado'].round(2) # Redondear a 2 decimales

# ==========================================
# 7. EXPORTACIÓN FINAL
# ==========================================
# Lista definitiva de columnas
cols_finales_ordenadas = [
    'titulo', 
    'empresa', 
    'pais', 
    'ciudad_limpia', 
    'ubicacion_raw',          
    'salario_medio_ppp_2024', 
    'indice_coste_vida_2024', 
    'indice_alquiler_2024',
    'salario_real_ajustado',
    'modalidad', 
    'es_teletrabajo', 
    'desc_longitud',          
    'url'
]

# Selección segura (solo columnas que existan)
cols_a_guardar = [c for c in cols_finales_ordenadas if c in df_final.columns]
df_final = df_final[cols_a_guardar]

# Guardar CSV
nombre_archivo = "Global Data Analyst Job Market 2025.csv"
ruta_salida = os.path.join(ruta_dataset, nombre_archivo)
df_final.to_csv(ruta_salida, index=False, sep=";", encoding="utf-8-sig")

print(f"\n[OK] Dataset generado: {nombre_archivo}")
print(f"[OK] Columnas incluidas: {list(df_final.columns)}")
print(f"[OK] Total filas: {len(df_final)}")