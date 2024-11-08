# Descrição
Este repositório contém um código em Python para realizar inferências em imagens utilizando o modelo YOLO, além de um Dockerfile para criar um ambiente isolado que executa o código em um container Docker. O código carrega um modelo YOLO pré-treinado, realiza a detecção de objetos em imagens de uma pasta específica e salva as imagens processadas em um diretório de saída. Abaixo, explicamos o funcionamento do código, como configurá-lo e como utilizar o Dockerfile para rodá-lo em um container.

# Estrutura do Repositório
O repositório contém os seguintes arquivos principais:

# imgTrain.py: 
- Código Python responsável pela detecção de objetos nas imagens utilizando o modelo YOLO.
# Dockerfile:
- Arquivo de configuração para criar um container Docker para rodar o código Python de forma isolada.

# Dependências
Antes de executar o código, as seguintes bibliotecas Python precisam estar instaladas:

1. **Python 3.7** ou superior.
2. **ultralytics**: Instale a biblioteca ULTRALYTICS utilizando o comando:
   ```bash
   pip install ultralytics===8.3.27
   ```
- **opencv-python-headless:** Para manipulação de imagens.
   ```bash
   pip install opencv-python-headless
   ```
- **matplotlib:** Para visualização das imagens (opcional).
  ```bash
   pip install matplotlib
   ```
