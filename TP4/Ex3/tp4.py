import nltk
import re
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
from sklearn import metrics
from sklearn.model_selection import cross_val_predict

# Ler o arquivo com os dados, mostre uma amostra do arquivo e exibe a contagem de cada uma das colunas.

dataset = pd.read_csv(r"S:\Sahra_Dados\ORI\TP4\Ex3\Tweets_Mg.csv", encoding='utf-8')
dataset.head()
dataset.count()

# Contar a quantidade de tweets para cada tipo: neutro, positivo e negativo.

dataset[dataset.Classificacao == 'Neutro'].count()
dataset[dataset.Classificacao == 'Positivo'].count()
dataset[dataset.Classificacao == 'Negativo'].count()

# Criar variáveis separadas para armazenar tweets e suas classificações.

tweets = dataset['Text'].values
classes = dataset['Classificacao'].values

#TF
print("\n--------------------Testes com TF----------------------")
vectorizer_tf = CountVectorizer(analyzer="word")
freq_tweets = vectorizer_tf.fit_transform(tweets)
modelo = MultinomialNB()
modelo.fit(freq_tweets,classes)
resultados1 = cross_val_predict(modelo, freq_tweets, classes, cv=10)
metrics.accuracy_score(classes,resultados1)
print(metrics.classification_report(classes,resultados1))
print (pd.crosstab(classes, resultados1, rownames=['Real'], colnames=['Predito'], margins=True))

#TF + bigram
print("\n---------------Testes com TF + 2-gram----------------")
vectorizer_tfbi = CountVectorizer(ngram_range=(2,2))
tf_bigram = vectorizer_tfbi.fit_transform(tweets)
modelo = MultinomialNB()
modelo.fit(tf_bigram,classes)
resultados2 = cross_val_predict(modelo, tf_bigram, classes, cv=10)
metrics.accuracy_score(classes,resultados2)
print(metrics.classification_report(classes,resultados2))
print (pd.crosstab(classes, resultados2, rownames=['Real'], colnames=['Predito'], margins=True))

#TF-IDF + 1-gram
print("\n---------------Testes com TF-IDF + 1-gram--------------")
vectorizer_tfidf1 = TfidfVectorizer(ngram_range=(1, 1))
tfidf1_tweets = vectorizer_tfidf1.fit_transform(tweets)
modelo = MultinomialNB()
modelo.fit(tfidf1_tweets, classes)
resultados3 = cross_val_predict(modelo, tfidf1_tweets, classes, cv=10)
metrics.accuracy_score(classes, resultados3)
print(metrics.classification_report(classes, resultados3))
print(pd.crosstab(classes, resultados3, rownames=['Real'], colnames=['Predito'], margins=True))

#TF-IDF + bigram
print("\n--------------Testes com TF-IDF + 2-gram--------------")
vectorizer_tfidf = TfidfVectorizer(ngram_range=(2, 2))
tfidf_tweets = vectorizer_tfidf.fit_transform(tweets)
modelo = MultinomialNB()
modelo.fit(tfidf_tweets, classes)
resultados3 = cross_val_predict(modelo, tfidf_tweets, classes, cv=10)
metrics.accuracy_score(classes, resultados3)
print(metrics.classification_report(classes, resultados3))
print(pd.crosstab(classes, resultados3, rownames=['Real'], colnames=['Predito'], margins=True))

# Árvore de Decisão
print("\n------------------Árvore de Decisão-------------------")
vectorizer_arvore = CountVectorizer()
freq_tweets_arvore = vectorizer_arvore.fit_transform(tweets)
modelo_arvore = DecisionTreeClassifier()
modelo_arvore.fit(freq_tweets_arvore, classes)
resultados_arvore = cross_val_predict(modelo_arvore, freq_tweets_arvore, classes, cv=10)
metrics.accuracy_score(classes, resultados_arvore)
print(metrics.classification_report(classes, resultados_arvore))
print(pd.crosstab(classes, resultados_arvore, rownames=['Real'], colnames=['Predito'], margins=True))

# Random Forest
print("\n-------------------Random Florest-------------------")
vectorizer_rf = CountVectorizer()
freq_tweets_rf = vectorizer_rf.fit_transform(tweets)
modelo_rf = RandomForestClassifier()
modelo_rf.fit(freq_tweets_rf, classes)
resultados_rf = cross_val_predict(modelo_rf, freq_tweets_rf, classes, cv=10)
metrics.accuracy_score(classes, resultados_rf)
print(metrics.classification_report(classes, resultados_rf))
print(pd.crosstab(classes, resultados_rf, rownames=['Real'], colnames=['Predito'], margins=True))

# Support Vector Machine (SVM)
print("\n------------Support Vector Machine (SVM)-------------")
vectorizer_svm = CountVectorizer()
freq_tweets_svm = vectorizer_svm.fit_transform(tweets)
modelo_svm = SVC()
modelo_svm.fit(freq_tweets_svm, classes)
resultados_svm = cross_val_predict(modelo_rf, freq_tweets_svm, classes, cv=10)
metrics.accuracy_score(classes, resultados_svm)
print(metrics.classification_report(classes, resultados_svm))
print(pd.crosstab(classes, resultados_svm, rownames=['Real'], colnames=['Predito'], margins=True))