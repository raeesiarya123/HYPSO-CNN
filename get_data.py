import os
import requests
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from webdriver_manager.chrome import ChromeDriverManager
from bs4 import BeautifulSoup

# Angi Chrome-binærens sti
chrome_path = "/usr/bin/google-chrome"

# Konfigurer Chrome-alternativer
chrome_options = Options()
chrome_options.binary_location = chrome_path
chrome_options.add_argument("--no-sandbox")
chrome_options.add_argument("--disable-dev-shm-usage")
chrome_options.add_argument("--headless")  # Valgfritt: Kjør uten GUI

# Start Chrome WebDriver
service = Service(ChromeDriverManager().install())
driver = webdriver.Chrome(service=service, options=chrome_options)

# Hoved-URL
base_url = "http://129.241.2.147:8009/"

# Opprett hovedmappen "collected_data" hvis den ikke finnes
base_save_dir = "raw_data"
os.makedirs(base_save_dir, exist_ok=True)

# 1️⃣ Hent directories fra hovedsiden
driver.get(base_url)
soup = BeautifulSoup(driver.page_source, "html.parser")
directories = [a.text.strip() for a in soup.find_all("a")]

# Lagringsstrukturer
subdirectories = []
png_files = {}
bip_extra_files = {}
bip_files = {}
extra_hsi0_files = {}

# 2️⃣ Gå inn i første nivå undermapper (hoved-mapper)
for directory in directories:
    first_level_url = base_url + directory
    driver.get(first_level_url)
    soup = BeautifulSoup(driver.page_source, "html.parser")

    # Finn undermapper inne i hovedmappen
    subdirs = [a.text.strip() for a in soup.find_all("a")]

    # Opprett mappe for denne dataset-mappen
    dataset_dir = os.path.join(base_save_dir, directory.strip("/"))
    os.makedirs(dataset_dir, exist_ok=True)

    # 3️⃣ Gå inn i undermapper for å hente PNG- og ".bip@"-filer
    for subdir in subdirs:
        second_level_url = first_level_url + subdir
        driver.get(second_level_url)
        soup = BeautifulSoup(driver.page_source, "html.parser")

        # Hent PNG-filer
        pngs = [a.text.strip() for a in soup.find_all("a") if a.text.strip().lower().endswith(".png")]
        for file in pngs:
            file_url = second_level_url + file
            save_path = os.path.join(dataset_dir, file)
            response = requests.get(file_url)
            with open(save_path, "wb") as f:
                f.write(response.content)
                print(f"Lastet ned: {file}")

        # Hent ".bip@"-filer
        bip_at_files = [a.text.strip() for a in soup.find_all("a") if a.text.strip().lower().endswith(".bip@")]
        for file in bip_at_files:
            file_url = second_level_url + file[:-1]
            save_path = os.path.join(dataset_dir, file)
            response = requests.get(file_url)
            with open(save_path, "wb") as f:
                f.write(response.content)
                print(f"Lastet ned: {file}")

        # Sjekk etter "hsi0"-mapper
        hsi0_subdirs = [a.text.strip() for a in soup.find_all("a") if "hsi0" in a.text.strip()]
        for hsi0_subdir in hsi0_subdirs:
            hsi0_url = second_level_url + hsi0_subdir
            driver.get(hsi0_url)
            soup = BeautifulSoup(driver.page_source, "html.parser")

            # Opprett en mappe for 'hsi0' data
            hsi0_dir = os.path.join(dataset_dir, hsi0_subdir.strip("/"))
            os.makedirs(hsi0_dir, exist_ok=True)

            # Hent .bip-filer
            bip_files_in_hsi0 = [a.text.strip() for a in soup.find_all("a") if a.text.strip().lower().endswith(".bip")]
            for file in bip_files_in_hsi0:
                file_url = hsi0_url + file
                save_path = os.path.join(hsi0_dir, file)
                response = requests.get(file_url)
                with open(save_path, "wb") as f:
                    f.write(response.content)
                    print(f"Lastet ned: {file}")

# Lukk nettleseren
driver.quit()

print("\n✅ Alle filer er lastet ned og lagret i 'raw_data/'!")