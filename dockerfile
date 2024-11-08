# Escolha a imagem base com Python
FROM python:3.12.5

# Instale as dependências necessárias
RUN pip install ultralytics==8.3.27 
RUN pip install opencv-python-headless 
RUN pip install matplotlib

ENV BASE_PATH = "/app"

# Defina variáveis de ambiente para o container
ENV MODELO="best01.pt"
ENV CAMINHO_IMAGEM = 
ENV PASTA_SALVAR="volumeTrain"

# Crie a pasta para salvar as imagens
RUN mkdir -p /app/volumeTrain

# Copie o código e os arquivos para o container
COPY /app /app

# Defina o diretório de trabalho
WORKDIR /app

# Execute o script ao iniciar o container
CMD ["python", "imgTrain.py"]
