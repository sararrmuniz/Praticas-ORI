import nltk
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn import metrics
from sklearn.model_selection import cross_val_predict
from nltk.corpus import stopwords

# Baixar a lista de stopwords em português
nltk.download('stopwords')

# Obter as stopwords em português
stopwords = set(stopwords.words('portuguese'))

# Ler o arquivo com os dados, mostre uma amostra do arquivo e exibe a contagem de cada uma das colunas.
dataset = pd.read_csv(r"S:\Sahra_Dados\ORI\TP4\Ex5\reforma_previdencia_rotulado.csv", encoding='utf-8', delimiter=';')
dataset.head()
dataset.count()

# Contar a quantidade de tweets para cada tipo: neutro, positivo e negativo.
dataset[dataset.Classificação == 'Neutro'].count()
dataset[dataset.Classificação == 'Positivo'].count()
dataset[dataset.Classificação == 'Negativo'].count()

# Criar variáveis separadas para armazenar tweets e suas classificações.
tweets = dataset['Tweet'].values
classes = dataset['Classificação'].values

# Remover as stopwords dos tweets
tweets = [' '.join([palavra for palavra in tweet.split() if palavra.lower() not in stopwords]) for tweet in tweets]

# Support Vector Machine (SVM)
print("\n------------Support Vector Machine (SVM)-------------")
vectorizer_svm = CountVectorizer()
freq_tweets_svm = vectorizer_svm.fit_transform(tweets)
modelo_svm = SVC()
modelo_svm.fit(freq_tweets_svm, classes)
resultados_svm = cross_val_predict(modelo_svm, freq_tweets_svm, classes, cv=10)
metrics.accuracy_score(classes, resultados_svm)
print(metrics.classification_report(classes, resultados_svm))
print(pd.crosstab(classes, resultados_svm, rownames=['Real'], colnames=['Predito'], margins=True))