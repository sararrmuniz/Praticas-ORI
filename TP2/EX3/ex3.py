import unidecode
import os
import string
import math
import time

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

# MAIN
t1_voc = time.time()# Obter o tempo antes da execução da função normaliza_documento()

path = r"S:\Sahra_Dados\ORI\TP2\EX3\musicas"
output_path = r"S:\Sahra_Dados\ORI\TP2\EX3\vocabulario.txt"

with open(output_path, "w") as vocabulario:
    documentos = normaliza_documentos(path)

    for termo in sorted(palavras_unicas):
        vocabulario.write(f"{termo}\n")

    t2_voc = time.time() #Obtem o tempo depois da execução da função normaliza_documento()


with open(output_path, "r") as vocabulario:
    vocabulario = vocabulario.read().splitlines()
    documentos = normaliza_documentos(path)

    idf = calcula_peso_idf(documentos, vocabulario)
    contador = 1

    t1_tfidf = time.time() #Obtem o tempo antes da execução da função calcula_peso_idf

    for doc in documentos:
        tf = calcula_peso_tf(doc, vocabulario)
        tfidf = calcula_peso_tfidf(tf, idf)

        t2_tfidf = time.time() #Obtem o tempo depois da execução da função caucula_peso_tfidf

        print(f"TF-IDF do documento {contador}: {tfidf}")
        contador += 1



# a)
count = 0
with open("vocabulario.txt", "r") as arquivo:
    for linha in arquivo:
        count += 1

print(f'O vocabulário possui {count} termos')

# b)

contador = 1

for doc in documentos:
        tf = calcula_peso_tf(doc, vocabulario)
        tfidf = calcula_peso_tfidf(tf, idf)

        max_tfidf = max(tfidf.values()) #max encontra o maior valor de tf-idf
        max_termo = max(tfidf, key=tfidf.get)

        print(f"Termo com maior valor de TF-IDF no documento {contador}: {max_termo} ({max_tfidf})")

        contador += 1

#c)
tempo_exe = t2_voc - t1_voc
print(f"Tempo para criar o vocabulário: {tempo_exe:.2f} segs")

#d)
tempo_exe = t2_tfidf - t1_tfidf
print(f"Tempo para calcular o TF-IDF da coleção: {tempo_exe:.2f} segs")