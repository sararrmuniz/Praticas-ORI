import unidecode
import os
import string
import math

palavras_unicas = set()  # conjunto vazio para armazenar palavras sem repetição

def remove_pontuacao(texto):
    """Remove a pontuação do texto e retorna o texto sem pontuação."""
    return texto.translate(str.maketrans('', '', string.punctuation))


def normaliza_palavras(texto):
    """Remove acentos e coloca as palavras em minúsculas."""
    texto_sem_acentos = unidecode.unidecode(texto)
    return texto_sem_acentos.lower()

def normaliza_documentos(file_path):
    """Função que trata os documentos."""

    documentos = []
    for file_name in os.listdir(file_path):
        if file_name.endswith(".txt"):
            with open(os.path.join(file_path, file_name), 'r', encoding='utf-8') as f:
                texto = f.readlines()

                palavras_documento = []

                for linha in texto:
                    texto_sem_pontuacao = remove_pontuacao(linha) # ignora pontuação no texto
                    palavras = sorted(texto_sem_pontuacao.split())  # separa e ordena as palavras

                    for palavra in palavras:
                        palavra_normalizada = normaliza_palavras(palavra)  # transforma palavras com letras maiúsculas em minúscula
                        palavras_documento.append(palavra_normalizada)

                        palavras_unicas.add(palavra_normalizada)  # adiciona cada palavra normalizada pelo método unidecode ao conjunto de palavras que não se repetem

                documentos.append(" ".join(palavras_documento)) #junta todas as palavras de um documento em uma única string

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
    """Função que calcula o IDF de um vocabulário"""
    freq_termos = {termo: 0 for termo in vocabulario} #dicionário em que cada termo do vocabulario inicia com zero.
    for doc in documentos:
        for termo in set(doc.split()): #para cada documento é criado um conjunto de palavras únicas, evitando a contagem repetida de termos
                            
            if termo in freq_termos: #verifica se cada palavra está presente no dicionário
                freq_termos[termo] += 1

    idf = {}
    n_docs = len(documentos) #o função len retorna a quantidade de documentos presente no diretório
    for termo in vocabulario:
        freq = freq_termos[termo]
        idf[termo] = math.log2(n_docs / freq) if freq > 0 else 0

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

        # Calcular o grau de similaridade usando a fórmula do cosseno
        similaridade = produto_escalar / (norma_vetor_consulta * norma_vetor_documento)

        similaridades[i+1] = similaridade

    return similaridades


# MAIN
diretorio_docs = r"S:\Sahra_Dados\ORI\TP3\Ex2\arquivos"
arq_vocabulario = r"S:\Sahra_Dados\ORI\TP3\Ex2\vocabulario.txt"
consulta = "to do"

#Cria e escreve no arquivo vocabularo.txt
with open(arq_vocabulario, "w") as vocabulario:
    documentos = normaliza_documentos(diretorio_docs)

    for termo in sorted(palavras_unicas):
        vocabulario.write(f"{termo}\n")

sim = calcular_similaridade_modelo_vetorial(arq_vocabulario, diretorio_docs, consulta)

# Exibir os resultados
for documento, similaridade in sim.items():
    print("Documento:", documento, "Similaridade:", similaridade)