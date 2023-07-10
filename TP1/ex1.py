import unidecode
import string
#1)
with open('texto1.txt', 'r') as arquivo: #Abri o arquivo com o open
    texto = arquivo.readlines() #lê as linhas presentes no arquivo

    palavras_unicas = set() #cria um conjunto vazio para armazenar as palavras únicas

for linha in texto:
    texto_sem_pontuacao = linha.translate(str.maketrans('', '', string.punctuation)) #ignora pontuação
    palavras = texto_sem_pontuacao.split() #divide cada linha em palavras usando o método split()
  
    for palavra in palavras:
        palavra_lower = palavra.lower() #trasforma palavras maiúsculas em minúsculas
        palavra_normalizada = unidecode.unidecode(palavra_lower) #retira acentuação das palavras
        palavras_unicas.add(palavra_normalizada) #Adiciona cada palavra normalizada ao conjunto de palavras únicas usando o método add().

palavras_ordenadas = sorted(palavras_unicas) #Ordena o conjunto de palavras usando a função sorted()

for palavra in palavras_ordenadas:
    print(palavra)
