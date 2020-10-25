#! /bin/python3

import pandas as pd
import re
from unidecode import unidecode
import nltk
nltk.download('stopwords')
import matplotlib.pyplot as plt
from gensim.models import KeyedVectors
from tensorflow import keras
import numpy as np

# Corpus da B2W, sem cortes
print('\n Importando aquivo B2W-Reviews01.csv...')
b2wCorpus = pd.read_csv("B2W-Reviews01.csv",";",usecols=['review_text','overall_rating'])

##
## PRE-PROCESSAMENTO
##

# Deixar o corpus somente com estrelas de 1-5
d = b2wCorpus.index[b2wCorpus["overall_rating"] < 1].tolist()
b2wCorpus=b2wCorpus.drop(b2wCorpus.index[d])
d = b2wCorpus.index[b2wCorpus["overall_rating"] > 5].tolist()
b2wCorpus=b2wCorpus.drop(b2wCorpus.index[d])

# Cleanning & Stopwords
text_cleaning_re = "@\S+|https?:\S+|http?:\S|[^A-Za-z0-9]+"
stop_words = nltk.corpus.stopwords.words('portuguese')
def cleanning(text, stem=False):
  text = unidecode(text)
  text = re.sub(text_cleaning_re, ' ', str(text).lower()).strip()
  text = re.sub("\d+", "", text)
  text = re.sub(r'(?:^| )\w(?:$| )', ' ', text).strip()
  tokens = []
  for token in text.split():
    if token not in stop_words:
        tokens.append(token)
  return " ".join(tokens)
  
print('\n Limpando Corpus (review_text)...')  
b2wCorpus.review_text = b2wCorpus.review_text.apply(lambda x: cleanning(x))

print("\n### x: review_text ###\n")
print(b2wCorpus.head)

print("\n### y: overall_rating ###\n")
print(b2wCorpus.overall_rating .value_counts().sort_index())

# histograma de palavras x qtd de linhas

print('\nSalvando o histograma de palavras (palavras_por_frase_histograma.png)...\n')

Words = [len(linha.split()) for linha in b2wCorpus["review_text"] if len(linha.split()) <=100 ]
plt.style.use('ggplot')
plt.title('Histograma de Palavras')
plt.legend()
plt.hist(Words, bins=[0,5,10,15,20,25,30,35,40,45,50,55,60,65,70,75,80,85], histtype='bar',align='mid', color='c',edgecolor='black',label='palavras/frase')
plt.savefig('palavras_por_frase_histograma.png')

##
## CODIFICACAO / Embedding
##

print('\nCodificado corpus, padding e insercao de coluna de tamanho de frase....\n')

N =  200001

# Arquivo original do NILC
with open("cbow_s50.txt", "r",encoding='utf-8') as file:
    head = [next(file) for x in range(N)]

head[0] = str(N-1)+ " " + "50"+ "\n" # Conserta contagem de palavras
with open("word2vec_200k.txt", "w",encoding='utf-8') as file:
    for line in head:
        file.write(line)

def vocaIndex(lista, stem=False):
    for indice in range(len(lista)):
        text=lista[indice].lower()
        if text in model.vocab:
             lista[indice] = model.vocab[text].index
        else: 
             # seta a palavra nao encontrada como a menos provavel
             # nao usei zero para não confundir com pad
             lista[indice] = '199999'
    return lista

model = KeyedVectors.load_word2vec_format('word2vec_200k.txt')

# Codificação usando NILC
def codifica(text, stem=False):
    tokens = nltk.word_tokenize(text)
    tokens = vocaIndex(tokens)
    return tokens

b2wCorpus.review_text = b2wCorpus.review_text.apply(lambda x: codifica(x))

# Cria coluna que será utilizada para ordenar por tamanho de palavras
b2wCorpus['ord'] = b2wCorpus.apply(lambda row: len(row.review_text), axis=1)
b2wCorpus['overall_rating'] = b2wCorpus.overall_rating.apply(lambda x: x - 1)

# Filtra somente frases com  10<tamanho do palavras<60
b2wCorpus = b2wCorpus.drop(b2wCorpus[b2wCorpus.ord < 10].index)
b2wCorpus = b2wCorpus.drop(b2wCorpus[b2wCorpus.ord > 60].index)
b2wCorpus.reset_index(drop=True, inplace=True)

# Padding
b2wCorpus.review_text = keras.preprocessing.sequence.pad_sequences(b2wCorpus.apply(lambda row: np.reshape(row.review_text,(-1)), axis=1), maxlen=60, padding='post').tolist()

print(b2wCorpus.head)

##
## Split de treino, validacao e teste
##

# Função de split
def train_validate_test_split(df, train_percent=.65, validate_percent=.1, seed=42):
    np.random.seed(seed)
    perm = np.random.permutation(df.index)
    m = len(df.index)
    train_end = int(train_percent * m)
    validate_end = int(validate_percent * m) + train_end
    train = df.iloc[perm[:train_end]]
    validate = df.iloc[perm[train_end:validate_end]]
    test = df.iloc[perm[validate_end:]]
    return train, validate, test

# Split
b2wCorpusTrain, b2wCorpusValidate, b2wCorpusTest = train_validate_test_split(b2wCorpus)

pri
b2wCorpusTrain=b2wCorpusTrain.reindex(b2wCorpusTrain['ord'].sort_values(ascending=False).index)
b2wCorpusTrain.head()

b2wCorpusValidate=b2wCorpusValidate.reindex(b2wCorpusValidate['ord'].sort_values(ascending=False).index)
b2wCorpusValidate.head

b2wCorpusTest=b2wCorpusTest.reindex(b2wCorpusTest['ord'].sort_values(ascending=False).index)
b2wCorpusTest.head

