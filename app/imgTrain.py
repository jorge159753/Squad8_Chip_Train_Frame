import os
import json
import yaml
from ultralytics import YOLO
import cv2
import matplotlib.pyplot as plt

# Definindo as variáveis
MODELO = "best01.pt"
CAMINHO_IMAGEM = os.path.join("imgTeste")
PASTA_SALVAR_IMAGENS = "volumeTrain/Imagens"
PASTA_SALVAR_LABELS = "volumeTrain/Labels"
DATA_YAML = "data.yaml"  # Caminho para o arquivo data.yaml

# Carregar as classes do arquivo data.yaml
def carregar_classes(yaml_path):
    try:
        with open(yaml_path, 'r') as f:
            data = yaml.safe_load(f)
            return data['names']  # Retorna a lista de classes
    except Exception as e:
        print(f"Erro ao carregar o arquivo YAML: {e}")
        return []

# Carrega as classes definidas no data.yaml
classes = carregar_classes(DATA_YAML)
if not classes:
    print("Erro: Não foi possível carregar as classes do arquivo YAML.")
    exit()

# Carrega o modelo YOLO
try:
    model = YOLO(MODELO)
    print("Modelo carregado com sucesso.")
except Exception as e:
    print(f"Erro ao carregar o modelo: {e}")
    exit()

# Verifica se o caminho da pasta de imagens existe
if os.path.exists(CAMINHO_IMAGEM):
    # Cria as pastas de salvamento, se não existirem
    os.makedirs(PASTA_SALVAR_IMAGENS, exist_ok=True)
    os.makedirs(PASTA_SALVAR_LABELS, exist_ok=True)

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
                resultados = model.predict(imagem)
                resultado = resultados[0]  # Primeiro resultado, se houver mais
                
                # Exibe as caixas delimitadoras e rótulos
                imagem_bbox = resultado.plot()
                
                # Cria a estrutura de anotações no formato solicitado
                anotacoes = []
                for det in resultado.boxes.data.tolist():
                    x1, y1, x2, y2, conf, cls = det  # Coordenadas, confiança, e classe
                    index_classe = int(cls)  # Índice da classe
                    if 0 <= index_classe < len(classes):
                        label = index_classe  # Usar o índice da classe
                        # Normalizando as coordenadas para a imagem (se necessário)
                        width, height = imagem.shape[1], imagem.shape[0]
                        x1_norm = x1 / width
                        y1_norm = y1 / height
                        x2_norm = x2 / width
                        y2_norm = y2 / height
                        anotacoes.append(f"{label} {conf:.6f} {x1_norm:.6f} {y1_norm:.6f} {x2_norm:.6f} {y2_norm:.6f}")
                        print(f"Detecção: Classe={label}, Confiança={conf:.2f}, Coordenadas=({x1_norm}, {y1_norm}, {x2_norm}, {y2_norm})")
                
                
                
                # Exibe o sucesso da inferência
                print(f"Inferência em '{caminho_completo}' realizada com sucesso.")
                
                # Salva a imagem processada na pasta de imagens
                caminho_salvar_imagem = os.path.join(PASTA_SALVAR_IMAGENS, nome_arquivo)
                cv2.imwrite(caminho_salvar_imagem, imagem_bbox)
                print(f"Imagem salva em '{caminho_salvar_imagem}'.")
                
                # Salva as anotações no formato solicitado no arquivo .txt na pasta de labels
                nome_arquivo_txt = os.path.splitext(nome_arquivo)[0] + ".txt"
                caminho_salvar_txt = os.path.join(PASTA_SALVAR_LABELS, nome_arquivo_txt)
                with open(caminho_salvar_txt, "w") as arquivo_txt:
                    arquivo_txt.write("\n".join(anotacoes))
                print(f"Anotações salvas em '{caminho_salvar_txt}'.")
                
                
            except Exception as e:
                print(f"Erro durante a inferência em '{caminho_completo}': {e}")
else:
    print("Erro: Caminho da pasta de imagens não encontrado.")
