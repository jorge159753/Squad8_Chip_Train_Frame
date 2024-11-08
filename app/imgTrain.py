import os
from ultralytics import YOLO
import cv2
import matplotlib.pyplot as plt

# Definindo as variáveis
MODELO = "best01.pt"
CAMINHO_IMAGEM = 
PASTA_SALVAR = "volumeTrain"


# Carrega o modelo YOLO
try:
    model = YOLO(MODELO)
    print("Modelo carregado com sucesso.")
except Exception as e:
    print(f"Erro ao carregar o modelo: {e}")

# Verifica se o caminho da pasta de imagens existe
if os.path.exists(CAMINHO_IMAGEM):

    # Itera sobre os arquivos de imagem na pasta
    for nome_arquivo in os.listdir(CAMINHO_IMAGEM):
        caminho_completo = os.path.join(CAMINHO_IMAGEM, nome_arquivo)
        
        # Verifica se é um arquivo
        if os.path.isfile(caminho_completo):
            try:
                # Carrega a imagem
                imagem = cv2.imread(caminho_completo)
                
                if imagem is None:
                    raise FileNotFoundError(f"A imagem '{caminho_completo}' não foi encontrada ou não pode ser aberta.")
                
                # Realiza a inferência com o modelo YOLO
                resultado = model.predict(imagem)
                imagem_bbox = resultado[0].plot()
                
                # Converte para RGB antes de salvar ou exibir (opcional)
                imagem_rgb = cv2.cvtColor(imagem_bbox, cv2.COLOR_BGR2RGB)
                
                # Exibe o sucesso da inferência
                print(f"Inferência em '{caminho_completo}' realizada com sucesso.")
                
                # Salva a imagem processada
                nome_arquivo_salvar = os.path.basename(caminho_completo)
                caminho_salvar = os.path.join(PASTA_SALVAR, nome_arquivo_salvar)
                
                # Cria a pasta de salvamento, se não existir
                os.makedirs(PASTA_SALVAR, exist_ok=True)
                
                cv2.imwrite(caminho_salvar, imagem_bbox)
                print(f"Imagem salva em '{caminho_salvar}'.")

            except Exception as e:
                print(f"Erro durante a inferência em '{caminho_completo}': {e}")
else:
    print("Erro: Caminho da pasta de imagens não encontrado.")