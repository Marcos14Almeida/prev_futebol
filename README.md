# Previsão de resultados de partidas com random forest

## Descrição do Projeto 

Através do uso do método de classificação random forest, tento prever o resultado de partidas futuras.

## Como usar o Projeto 

Através do uso de Python, eu uso um dataset com informações a respeito dos times, clube mandante e retrospecto das últimas 5 partidas. A partir disso, eu filtro o dataset em train_set e test_set aplico feature scaling e o método de random forest no train_set do sklearn para obter uma previsão dos resultados. Modificando os hiperparametros e observando as métricas de Confusion Matrix, Recall, Accuracy e F1-Score, o algoritmo foi aperfeiçoado para melhores resultados.

Ao final, é gerado um gráfico 3d comparando o resultado real com o esperado, e com um outro dataset de teste de uma nova rodada do Brasileirão é gerado as previsões.

O algoritmo ainda é impreciso, tanto devido a imprevisibilidade dos resultados quanto pelo dataset pequeno com poucas informações.

## Como Executar o projeto

Baixe os arquivos e com o uso de algum compilador python com suporte para Matplotlib, Pandas e Sklearn instalados execute o código do arquivo ".py".

## Screenshot

<p align="center">
  <img src="https://github.com/Marcos14Almeida/prev_futebol/blob/master/result.jpg" width="200" title="Screenshot">
  </a>
</p>
