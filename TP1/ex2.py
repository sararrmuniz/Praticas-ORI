#2)
import string
import unidecode

#FUNÇÕES

def processa_arquivo(nome_arquivo): #função feita para criar o vocabulario
    # abre o arquivo para leitura
    with open(nome_arquivo, 'r') as arquivo: #abre o arquivo
        texto = arquivo.read() # lê todo o conteúdo do arquivo

        texto_sem_pontuacao = texto.translate(str.maketrans('', '', string.punctuation)) #retira a pontuação do texto
        palavras_minusculas = texto_sem_pontuacao.lower()  #transforma as palavras maiúsculas em minúsculas
        palavra_normalizada = unidecode.unidecode(palavras_minusculas) #remove acentuação
        palavras = palavra_normalizada.split() #divide cada linha em palavras usando o método split()
        palavras_unicas = list(set(palavras)) #remove palavras repetidas fazendo com que apareçam uma única vez
        palavras_unicas.sort() #ordena em ordem alfabética

        return palavras_unicas #retorna o vocabulario

def bag_of_words(vocabulario, arquivo): #função criada para retornar a bag of words de um vocabulário
    with open(arquivo, 'r') as arquivo: #abre o arquivo
        texto = arquivo.read().lower() #lê o arquivo e transforma as palavras maiúsculas em minúsculas
        texto = texto.translate(str.maketrans('', '', string.punctuation)) #remove pontuação
      
    bag_of_words = [1 if palavra in texto else 0 for palavra in vocabulario] #verifica as condições da bag of words
    return bag_of_words #retorna a bag of words

#MAIN
vocabulario = processa_arquivo('texto1.txt') #cria o vocabulário
bagofwords = bag_of_words(vocabulario,'documento.txt') #retorna a bag of words de acordo com o vocabulário e o documento especificado
print(bagofwords)
