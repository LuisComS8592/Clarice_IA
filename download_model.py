import gdown
import zipfile
import os
import shutil

# --- CONFIGURAÇÃO ---
FILE_ID = "1z_kDRIEbuV06yAPg9MTDYRYnuvMHNDmU" 

OUTPUT_FILENAME = "model.zip"
DESTINATION_FOLDER = "./model"

def download_and_extract():
    print("="*50)
    print("📥 INICIANDO DOWNLOAD DO MODELO CLARICE")
    print("="*50)

    # 1. Baixar o arquivo
    url = f'https://drive.google.com/uc?id={FILE_ID}'
    
    # Se o arquivo já existe, remove para evitar corrupção
    if os.path.exists(OUTPUT_FILENAME):
        os.remove(OUTPUT_FILENAME)

    print(f"Baixando de: {url}...")
    gdown.download(url, OUTPUT_FILENAME, quiet=False, fuzzy=True)

    # 2. Criar pasta de destino
    if not os.path.exists(DESTINATION_FOLDER):
        os.makedirs(DESTINATION_FOLDER)
        print(f"📂 Pasta '{DESTINATION_FOLDER}' criada.")
    else:
        print(f"⚠️ A pasta '{DESTINATION_FOLDER}' já existe. Sobrescrevendo arquivos...")

    # 3. Extrair
    print("📦 Extraindo arquivos...")
    try:
        with zipfile.ZipFile(OUTPUT_FILENAME, 'r') as zip_ref:
            zip_ref.extractall(DESTINATION_FOLDER)
        print("✅ Extração concluída!")
        
        # 4. Limpeza (Opcional: apagar o zip depois)
        os.remove(OUTPUT_FILENAME)
        print("🧹 Arquivo temporário (.zip) removido.")

        print("\n🎉 TUDO PRONTO! O modelo está instalado em './modelo'.")
        print("Agora você pode rodar: streamlit run app.py")

    except zipfile.BadZipFile:
        print("❌ ERRO: O arquivo baixado não é um ZIP válido.")
        print("Verifique se o ID do Google Drive está correto e se o arquivo é público.")
    except Exception as e:
        print(f"❌ Ocorreu um erro: {e}")

if __name__ == "__main__":
    download_and_extract()
