import ssl
import certifi
import os
import sys
import subprocess
import platform
import re
import time
import random
import pandas as pd
from selenium.webdriver.common.by import By
from urllib.parse import quote_plus
try:
    import undetected_chromedriver as uc
except ImportError:
    print("Installing undetected-chromedriver...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "undetected-chromedriver"])
    except subprocess.CalledProcessError as install_error:
        raise RuntimeError(
            "No se pudo instalar automáticamente undetected-chromedriver. "
            "Instala el paquete manualmente con 'pip install undetected-chromedriver'."
        ) from install_error
    import undetected_chromedriver as uc


def detectar_version_chrome() -> int | None:
    """Devuelve la versión mayor de Chrome instalada (ej. 142) si se puede detectar."""
    version_env = os.getenv("CHROME_VERSION" ) or os.getenv("CHROME_VERSION_MAIN")
    if version_env:
        try:
            return int(version_env.split(".")[0])
        except ValueError:
            pass

    if platform.system() == "Windows":
        claves = [
            r"HKEY_CURRENT_USER\Software\Google\Chrome\BLBeacon",
            r"HKEY_LOCAL_MACHINE\Software\Google\Chrome\BLBeacon",
        ]
        for clave in claves:
            try:
                salida = subprocess.check_output(
                    ["reg", "query", clave, "/v", "version"],
                    encoding="utf-8",
                    errors="ignore",
                )
                coincidencia = re.search(r"(\d+\.\d+\.\d+\.\d+)", salida)
                if coincidencia:
                    return int(coincidencia.group(1).split(".")[0])
            except (subprocess.CalledProcessError, FileNotFoundError):
                continue
    return None
    import undetected_chromedriver as uc

# ==========================================
# 1. PARCHE SSL (Para Mac OS)
# ==========================================

ssl._create_default_https_context = ssl._create_unverified_context

# ==========================================
# 2. CONFIGURACIÓN DEL NAVEGADOR
# ==========================================

options = uc.ChromeOptions()
options.add_argument("--window-size=1920,1080")
options.add_argument("--disable-popup-blocking")
# options.add_argument("--headless") # Descomentar para modo oculto

driver_kwargs = {"options": options}
version_chrome = detectar_version_chrome()
if version_chrome:
    driver_kwargs["version_main"] = version_chrome
    print(f"Usando Chrome versión {version_chrome} detectada automáticamente")
else:
    print("No se pudo detectar la versión de Chrome; se usará la predeterminada de uc")

driver = uc.Chrome(**driver_kwargs)

# ==========================================
# 3. PARÁMETROS (MULTI-PAÍS)
# ==========================================

PAISES = {
    "ES": "https://es.indeed.com",      # España
    "US": "https://www.indeed.com",     # EE.UU.
    "UK": "https://uk.indeed.com",      # Reino Unido
    "DE": "https://de.indeed.com",      # Alemania
    "MX": "https://mx.indeed.com",      # México
    "FR": "https://fr.indeed.com",      # Francia
}

keyword = "data analyst"
location = "" 
max_pages_per_country = 5 

keyword_enc = quote_plus(keyword)
location_enc = quote_plus(location)

ofertas = []
ids_vistos = set()

# ==========================================
# 4. SCRAPING
# ==========================================

print("🌍 Iniciando scraping...")

try:
    for codigo_pais, dominio_base in PAISES.items():
        print(f"\n✈️  PROCESANDO PAÍS: {codigo_pais}")
        
        base_url = f"{dominio_base}/jobs?q={{}}&l={{}}&start={{}}"
        
        for page in range(0, max_pages_per_country * 10, 10):
            url = base_url.format(keyword_enc, location_enc, page)
            print(f"   📄 Página start={page}")
            
            driver.get(url)
            time.sleep(random.uniform(4, 6))

            # Cerrar Pop-ups
            try:
                close_btn = driver.find_element(By.CSS_SELECTOR, "button[aria-label='cerrar'], button[aria-label='close'], div[aria-label='Cerrar']")
                close_btn.click()
            except:
                pass 

            # Detectar Cloudflare
            if "challenge" in driver.title.lower():
                print("   ⚠️ Cloudflare detectado. Esperando 15s...")
                time.sleep(15)

            job_cards = driver.find_elements(By.CSS_SELECTOR, "div.job_seen_beacon")
            if not job_cards:
                job_cards = driver.find_elements(By.CSS_SELECTOR, "td.resultContent")

            print(f"      → Ofertas: {len(job_cards)}")

            if not job_cards:
                if page > 0: break
                else: continue

            for card in job_cards:
                try:
                    # --- ID ÚNICO ---
                    try:
                        job_id = card.get_attribute("data-jk")
                        if not job_id:
                            link_elem = card.find_element(By.CSS_SELECTOR, "a.jcs-JobTitle")
                            job_id = link_elem.get_attribute("data-jk")
                    except:
                        continue

                    if not job_id or job_id in ids_vistos:
                        continue
                    ids_vistos.add(job_id)

                    # --- EXTRACCIÓN ---
                    
                    # 1. Título
                    try:
                        title = card.find_element(By.CSS_SELECTOR, "h2.jobTitle span").text.strip()
                    except:
                        title = "Data Analyst"

                    # 2. Empresa
                    try:
                        company = card.find_element(By.CSS_SELECTOR, "span[data-testid='company-name']").text.strip()
                    except:
                        company = "Confidencial"

                    # 3. Ubicación (Texto sucio)
                    try:
                        location_txt = card.find_element(By.CSS_SELECTOR, "div[data-testid='text-location']").text.strip()
                    except:
                        location_txt = "Ubicación desconocida"

                    # 4. Longitud descripción (Numérica)
                    # Intentamos sacar el snippet, si falla, medimos toda la tarjeta para evitar 0s
                    try:
                        summary_elem = card.find_element(By.CSS_SELECTOR, "div.job-snippet")
                        summary_text = summary_elem.text.strip()
                        desc_len = len(summary_text)
                    except:
                        desc_len = len(card.text)

                    # --- VARIABLES DERIVADAS ---
                    
                    # Modalidad (Categórica Target)
                    loc_lower = location_txt.lower()
                    if "remoto" in loc_lower or "remote" in loc_lower:
                        modalidad = "Remoto"
                    elif "híbrido" in loc_lower or "hybrid" in loc_lower:
                        modalidad = "Híbrido"
                    else:
                        modalidad = "Presencial"

                    link = f"{dominio_base}/viewjob?jk={job_id}"

                    ofertas.append({
                        "titulo": title,
                        "empresa": company,
                        "pais": codigo_pais,
                        "ubicacion_raw": location_txt,
                        "modalidad": modalidad,
                        "desc_longitud": desc_len,
                        "url": link
                    })

                except Exception:
                    continue
            
            driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
            time.sleep(random.uniform(2, 4))
        
        print(f"✅ País {codigo_pais} terminado.")
        time.sleep(2)

except Exception as e:
    print(f"❌ Error: {e}")

finally:
    driver.quit()

# ==========================================
# 5. GUARDADO
# ==========================================

if ofertas:
    # Ruta relativa: ../dataset/desde_source
    ruta_source = os.path.dirname(os.path.abspath(__file__))
    ruta_proyecto = os.path.dirname(ruta_source)
    ruta_dataset = os.path.join(ruta_proyecto, "dataset")
    os.makedirs(ruta_dataset, exist_ok=True)
    
    ruta_archivo = os.path.join(ruta_dataset, "indeed_global_final.csv")
    
    df = pd.DataFrame(ofertas)
    df.to_csv(ruta_archivo, index=False, encoding="utf-8-sig")
    
    print("\n✅ Extracción finalizada.")
    print(f"📁 Guardado en: {ruta_archivo}")
    print(f"📊 Total registros: {len(df)}")
    print(df.head())
else:
    print("\n⚠️ No se encontraron datos.")