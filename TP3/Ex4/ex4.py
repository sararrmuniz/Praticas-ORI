import nltk
from nltk.corpus import stopwords
import unidecode
import string
import os
import string
import math

porter = nltk.PorterStemmer()

nltk.download('stopwords')

palavras_unicas = set()  #conjunto vazio para armazenar palavras sem repetição

def normaliza_texto(texto, stopwords_list):
    # Remover acentos
    texto = unidecode.unidecode(texto)

    # Remover números e pontuação
    texto = texto.translate(str.maketrans('', '', string.digits + string.punctuation))

    # Separar/dividir o texto em palavras
    termos = texto.split()

    # Remover stopwords
    stop_words = set([palavra.lower() for palavra in stopwords_list])
    termos = [palavra.lower() for palavra in termos if palavra.lower() not in stop_words]

    # Stemming
    termos_stem = [porter.stem(termo) for termo in termos]

    # Juntar as palavras novamente em um texto normalizado
    texto_normalizado = ' '.join(termos_stem)

    return texto_normalizado


def normaliza_documentos(file_path):
    """Função que trata os documentos."""

    stopwords_english = stopwords.words('english')

    documentos = []

    for file_name in os.listdir(file_path):
        if file_name.endswith(".txt"):
            with open(os.path.join(file_path, file_name), 'r', encoding='utf-8') as f:
                texto = f.read()  # Lê o texto completo do arquivo

                texto_normalizado = normaliza_texto(texto, stopwords_english)  #Normaliza o texto completo
                palavras_unicas.update(texto_normalizado.split())

                documentos.append(texto_normalizado)

    return documentos

def calcula_peso_tf(documento, vocabulario):
    """Função que calcula o TF  de um documento."""
    tf = {}
    for termo in vocabulario:
        tf[termo] = 0  # inicializa a contagem de termos com 0 ocorrencias

    for termo in documento.split():
        if termo in vocabulario:
            tf[termo] += 1  # incrementa a contagem de ocorrências para cada termo que aparece no documento

    for termo in vocabulario:
        if tf[termo] > 0:
            tf[termo] = 1 + math.log2(tf[termo])

    return tf

def calcula_peso_idf(documentos, vocabulario):
    freq_termos = {termo: 0 for termo in vocabulario}
    n_docs = len(documentos)
    for doc in documentos:
        for termo in set(doc.split()):
            if termo in freq_termos:
                freq_termos[termo] += 1

    idf = {}
    for termo in vocabulario:
        freq = freq_termos[termo]
        idf[termo] = math.log2(n_docs / (freq + 1))  # Adicionar 1 para evitar divisão por zero

    return idf


def calcula_peso_tfidf(tf, idf):
    """Função que calcula o TF-IDF de um documento"""
    tfidf = {}
    for termo in tf:
        if tf[termo] > 0:
            tfidf[termo] = tf[termo] * idf[termo]
        else:
            tfidf[termo] = 0.0
    return tfidf

def calcular_similaridade_modelo_vetorial(vocabulario, diretorio_documentos, consulta):
    documentos = normaliza_documentos(diretorio_documentos)
    vocabulario_lista = list(palavras_unicas)

    # Calcular os pesos IDF
    idf = calcula_peso_idf(documentos, vocabulario_lista)

    # Calcular o vetor da consulta
    vetor_consulta = calcula_peso_tf(consulta, vocabulario_lista)
    vetor_consulta_tfidf = calcula_peso_tfidf(vetor_consulta, idf)

    # Calcular o grau de similaridade para cada documento
    similaridades = {}

    for i, documento in enumerate(documentos):
        vetor_documento = calcula_peso_tf(documento, vocabulario_lista)
        vetor_documento_tfidf = calcula_peso_tfidf(vetor_documento, idf)

        # Calcular o produto escalar entre o vetor do documento e o vetor da consulta
        produto_escalar = sum(vetor_consulta_tfidf[termo] * vetor_documento_tfidf[termo] for termo in vetor_consulta_tfidf)

        # Calcular a norma dos vetores
        norma_vetor_consulta = math.sqrt(sum(vetor_consulta_tfidf[termo] ** 2 for termo in vetor_consulta_tfidf))
        norma_vetor_documento = math.sqrt(sum(vetor_documento_tfidf[termo] ** 2 for termo in vetor_documento_tfidf))

        # Checa divisão por zero
        if norma_vetor_consulta * norma_vetor_documento != 0:
            # Calcular o grau de similaridade usando a fórmula do cosseno
            similaridade = produto_escalar / (norma_vetor_consulta * norma_vetor_documento)
        else:
            similaridade = 0.0

        similaridades[i+1] = similaridade

    return similaridades

# MAIN
diretorio_docs = r"S:\Sahra_Dados\ORI\TP3\Ex4\musicas"
arq_vocabulario = r"S:\Sahra_Dados\ORI\TP3\Ex4\vocabulario.txt"

consultas = ["it's all about love", "forever young", "broken love", "believe yourself forever", "don't blame me for crying love"]

#Cria e escreve no arquivo vocabularo.txt
with open(arq_vocabulario, "w") as vocabulario:
    documentos = normaliza_documentos(diretorio_docs)

    for termo in sorted(palavras_unicas):
        vocabulario.write(f"{termo}\n")

for consulta in consultas:
    sim = calcular_similaridade_modelo_vetorial(arq_vocabulario, diretorio_docs, consulta)

    # Exibir os resultados para a consulta atual
    print("Consulta:", consulta)
    for documento, similaridade in sim.items():
        print("Documento:", documento, "Similaridade:", similaridade)
    print()  # Adiciona uma linha em branco entre as consultas

