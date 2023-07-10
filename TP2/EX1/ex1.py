import unidecode
import os
import string


def remove_pontuacao(texto):
    """Remove a pontuação do texto e retorna o texto sem pontuação."""
    return texto.translate(str.maketrans('', '', string.punctuation))


def normaliza_palavras(texto):
    """Remove acentos e coloca as palavras em minúsculas."""
    texto_sem_acentos = unidecode.unidecode(texto)
    return texto_sem_acentos.lower()


def cria_vocabulario(file_path):
    """Lê um arquivo de texto, remove a pontuação e normaliza as palavras.
    Retorna uma lista de palavras únicas no arquivo."""
    with open(file_path, 'r', encoding='utf-8') as f:
        texto = f.read()
        texto_sem_pontuacao = remove_pontuacao(texto)
        palavras_normalizadas = normaliza_palavras(texto_sem_pontuacao)
        palavras = palavras_normalizadas.split()
        palavras_unicas = list(set(palavras))
        return palavras_unicas


def cria_vocabulario_total(path):
    """Lê todos os arquivos de texto em um diretório e cria um vocabulário
    com todas as palavras únicas dos arquivos. Retorna o vocabulário."""
    os.chdir(path)

    vocabulario_total = []
    for file in os.listdir():
        if file.endswith(".txt"):
            file_path = os.path.join(path, file)
            palavras_do_arquivo = cria_vocabulario(file_path)
            vocabulario_total.extend(palavras_do_arquivo)

    vocabulario_total = list(set(vocabulario_total))
    vocabulario_total.sort()
    return vocabulario_total


def salva_vocabulario(vocabulario_total, output_path):
    """Salva o vocabulário em um arquivo de texto."""
    with open(output_path, 'w', encoding='utf-8') as f:
        for palavra in vocabulario_total:
            f.write(palavra + "\n")


def bag_of_words(vocabulary, file_paths):
    bags_of_words = []
    for file_path in file_paths:
        with open(file_path, 'r') as f:
            text = f.read().lower()
            text = text.translate(str.maketrans('', '', string.punctuation))
            bag_of_words = [1 if word in text else 0 for word in vocabulary]
            bags_of_words.append(bag_of_words)
    return bags_of_words


# MAIN
path = r"S:\Sahra_Dados\ORI\TP2\EX1\arquivos"
output_path = "vocabulario_total.txt"

vocabulario_total = cria_vocabulario_total(path)
salva_vocabulario(vocabulario_total, output_path)

file_paths = [os.path.join(path, file) for file in os.listdir(path) if file.endswith(".txt")]
bags_of_words = bag_of_words(vocabulario_total, file_paths)
print(bags_of_words)

