# Projeto de Classificação de Defeitos em Superfícies

Este projeto envolve o desenvolvimento de um pipeline completo para processar dados de imagem de um dataset de defeitos em superfícies, treinar um modelo de classificação usando PyTorch, e avaliar o desempenho do modelo com métricas detalhadas.

Dataset: https://www.kaggle.com/datasets/kaustubhdikshit/neu-surface-defect-database/code?datasetId=819567

## Instalar dependências
python -m venv env
.\env\Scripts\activate
pip install -r requirements.txt

## Estrutura do Projeto
- `train_pipe.py`: Script principal para treinamento e avaliação do modelo.
cls; python .\train_pipe.py --dataset 'projeto_neu_surface\dataset_neu\'
- `metrics.csv`: Arquivo gerado contendo as métricas de avaliação do modelo.
- `inference.py`: Script principal para inference
cls; python .\inference.py --model 'model.pth' --data 'validation\images\'

## Modelo pré-treinado disponível
O modelo pré-treinado model.pth está disponível para testes no seguinte link: https://drive.google.com/file/d/1iiJ19dArftXH8RinOP8RrhB_oFrCR7hR/view?usp=sharing