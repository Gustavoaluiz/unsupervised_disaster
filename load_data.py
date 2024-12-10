import os
import gdown
import zipfile

# URL do arquivo no Google Drive
url = "https://drive.google.com/uc?id=1on4_cbLyDDS3sXyllwpk7qzVwoZQg3kJ"

# Nome do arquivo a ser salvo
output = "dataset.zip" 

# Baixar o arquivo
gdown.download(url, output, quiet=False)

# Verificar se o arquivo é ZIP e descompactar
if zipfile.is_zipfile(output):
    with zipfile.ZipFile(output, 'r') as zip_ref:
        # Criar um diretório para extrair os arquivos
        extract_path = "extracted_data"
        os.makedirs(extract_path, exist_ok=True)
        zip_ref.extractall(extract_path)
        print(f"Arquivos extraídos para: {extract_path}")
else:
    print("O arquivo baixado não é um arquivo ZIP.")
