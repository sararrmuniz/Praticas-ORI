import nltk
import re
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn import metrics
from sklearn.model_selection import cross_val_predict
import os

# Ler o arquivo com os dados, mostre uma amostra do arquivo e exibe a contagem de cada uma das colunas.

dataset = pd.read_csv(r"S:\Sahra_Dados\ORI\TP4\Ex1\Tweets_Mg.csv", encoding='utf-8')
dataset.head()
dataset.count()

# Contar a quantidade de tweets para cada tipo: neutro, positivo e negativo.

dataset[dataset.Classificacao == 'Neutro'].count()
dataset[dataset.Classificacao == 'Positivo'].count()
dataset[dataset.Classificacao == 'Negativo'].count()

# Criar variáveis separadas para armazenar tweets e suas classificações.

tweets = dataset['Text'].values
classes = dataset['Classificacao'].values

# Usar TF-IDF para representação dos textos (1-gram)

vectorizer = TfidfVectorizer(ngram_range=(1, 1))
tfidf_tweets = vectorizer.fit_transform(tweets)

# Treinar o modelo usando o Multinomial Naive Bayes

modelo = MultinomialNB()
modelo.fit(tfidf_tweets, classes)

# Realizar testes manuais fornecendo alguns tweets como entrada para o modelo

testes = ['O governo de Minas é uma tragédia, muito ruim',
          'Estou muito feliz com o governo de Minas esse ano',
          'O estado de Minas Gerais decretou calamidade financeira!!!',
          'A segurança do estado está deixando a desejar',
          'O governador de Minas é do Novo']
print(testes)

# Calcular o TF-IDF para os tweets de teste

tfidf_testes = vectorizer.transform(testes)

# Fazer previsões para os tweets de teste usando o modelo treinado

modelo.predict(tfidf_testes)

# Usar validação cruzada para uma avaliação mais robusta do modelo

resultados = cross_val_predict(modelo, tfidf_tweets, classes, cv=10)

# Calcular a acurácia do modelo

metrics.accuracy_score(classes, resultados)

# Exibir o relatório de classificação

print(metrics.classification_report(classes, resultados))

# Exibir a matriz de confusão

print(pd.crosstab(classes, resultados, rownames=['Real'], colnames=['Predito'], margins=True))
