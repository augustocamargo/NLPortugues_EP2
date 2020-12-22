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
from tensorflow.keras.losses import sparse_categorical_crossentropy
from tensorflow.keras import layers
from keras.layers import Dense, Dropout, LSTM
import tensorflow as tf
import pickle
import os
from multiprocessing import Pool

##
## São necessários dois inputs para rodar este script: 2W-Reviews01.csv e cbow_s50.txt
##

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
def cleanning(text, stem=False):
  stop_words = nltk.corpus.stopwords.words('portuguese')
  text_cleaning_re = "@\S+|https?:\S+|http?:\S|[^A-Za-z0-9]+"
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
plt.clf()

# Tamanho máximo de frase escolhida
SEQUENCE_MAXLEN = 50

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

# Filtra somente frases com  10<tamanho do palavras<SEQUENCE_MAXLEN
b2wCorpus = b2wCorpus.drop(b2wCorpus[b2wCorpus.ord < 10].index)
b2wCorpus = b2wCorpus.drop(b2wCorpus[b2wCorpus.ord > SEQUENCE_MAXLEN].index)
b2wCorpus.reset_index(drop=True, inplace=True)

# Padding
b2wCorpus.review_text = keras.preprocessing.sequence.pad_sequences(b2wCorpus.apply(lambda row: np.reshape(row.review_text,(-1)), axis=1), maxlen=SEQUENCE_MAXLEN, padding='post').tolist()

print(b2wCorpus.head)

# Emb para camadas Keras
model = KeyedVectors.load_word2vec_format('word2vec_200k.txt')
emb = model.get_keras_embedding()

##
## Split de treino, validacao e teste
##

print('\nCriando os splits de Treino, Validacao e Teste.')

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

print('\n Dataset de Treino: 65%\n')
b2wCorpusTrain=b2wCorpusTrain.reindex(b2wCorpusTrain['ord'].sort_values(ascending=False).index)
print(b2wCorpusTrain.head)

print('\n Dataset de Validação: 10%\n')
b2wCorpusValidate=b2wCorpusValidate.reindex(b2wCorpusValidate['ord'].sort_values(ascending=False).index)
print(b2wCorpusValidate)

print('\n Dataset de Validação: 25%\n')
b2wCorpusTest=b2wCorpusTest.reindex(b2wCorpusTest['ord'].sort_values(ascending=False).index)
print(b2wCorpusTest.head)

# Criação das séries
# Treino, Validação e Teste
x_train =  [ emb for emb in b2wCorpusTrain.review_text]
y_train =   b2wCorpusTrain.overall_rating

x_val = [ emb for emb in b2wCorpusValidate.review_text ]
y_val = b2wCorpusValidate.overall_rating

x_test = [ emb for emb in b2wCorpusTest.review_text ]
y_test = b2wCorpusTest.overall_rating

x_train = np.asarray(x_train)
x_val =np.asarray(x_val)
x_test =np.asarray(x_test)

if os.path.exists("resultados.txt"):
    os.remove("resultados.txt")

##
## nn here go!
##
def myNet(SEQUENCE_MAXLEN,emb,nome,tipo,units,dropout,batch_size,epochs,x_train,y_train,x_val,y_val,x_test,y_test):
      
    if os.path.exists("weights.hdf5"):
        os.remove("weights.hdf5")

    model = keras.Sequential()
    model.add(layers.Input(shape=(SEQUENCE_MAXLEN, )))
    model.add(emb)
    if tipo == 'LSTM':
        model.add(keras.layers.LSTM(units,dropout=dropout))
        opt="adam"
    else:
        forward_layer = keras.layers.LSTM(units, activation='relu',dropout=dropout)
        backward_layer = keras.layers.LSTM(units, activation='relu', go_backwards=True,dropout=dropout)
        model.add(keras.layers.Bidirectional(forward_layer, backward_layer=backward_layer))
        model.add(keras.layers.Dropout(dropout))
        #opt = tf.keras.optimizers.SGD(learning_rate=.01, momentum=.9)
        opt="adam"
    model.add(keras.layers.Dense(5, activation='softmax'))
    model.compile(optimizer=opt,loss=sparse_categorical_crossentropy, metrics=["accuracy"])
    checkpointer = tf.keras.callbacks.ModelCheckpoint(filepath='weights' + '_ ' + nome + '_Units_' + str(units) + '_Dropouts_' + str(dropout) + '_Batchs_' + str(batch_size) + '.hdf5', verbose=1, save_best_only=True)
    es =  tf.keras.callbacks.EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=5)
    csv_logger = tf.keras.callbacks.CSVLogger('History_Log'  + '_' + nome + '_Units_' + str(units) + '_Dropouts_' + str(dropout) + '_Batchs_' + str(batch_size) +'.csv', append=True, separator=',')
    history = model.fit(
        x= x_train, y=y_train, batch_size=batch_size, epochs=epochs, validation_data=(x_val, y_val), callbacks=[checkpointer,es,csv_logger])

##
## It's all about the results!
##
    
    with open('History_Dump'  + '_' + nome + '_Units_' + str(units) + '_Dropouts_' + str(dropout) + '_Batchs_' + str(batch_size) +'.hist', 'wb') as h:
        pickle.dump(history.history, h)

    plt.title('Loss: ' + nome + ' - Units: ' + str(units) + ' - Dropouts: ' + str(dropout) + ' - Batchs: ' + str(batch_size))
    plt.xlabel('epochs')
    plt.ylabel('Loss')
    plt.plot(history.history['loss'], label='train')
    plt.plot(history.history['val_loss'], label='valid')
    plt.legend()
    plt.savefig('Loss'  + '_' + nome + '_Units_' + str(units) + '_Dropouts_' + str(dropout) + '_Batchs_' + str(batch_size) +'.png')
    plt.clf()

    plt.title('Accuracy: ' + nome + ' - Units: ' + str(units) + ' - Dropouts: ' + str(dropout) + ' - Batchs: ' + str(batch_size))
    plt.xlabel('epochs')
    plt.ylabel('accuracy')
    plt.plot(history.history['accuracy'], label='train')
    plt.plot(history.history['val_accuracy'], label='valid')
    plt.legend()
    plt.savefig('Accuracy' + '_ ' + nome + '_Units_' + str(units) + '_Dropouts_' + str(dropout) + '_Batchs_' + str(batch_size) +'.png')
    plt.clf()

    model.load_weights('weights' + '_ ' + nome + '_Units_' + str(units) + '_Dropouts_' + str(dropout) + '_Batchs_' + str(batch_size) + '.hdf5')
    scores = model.evaluate(x_test, y_test, verbose=1)
    titulo = nome + ' - Units: ' + str(units) + ' - Dropouts: ' + str(dropout) + ' - Batchs: ' + str(batch_size)
    acc = "Acuracia - %s: %.2f%%" % (titulo, scores[1]*100)
    print(acc)
    print('Score: ' + str(scores))
    with open('resultados' '_ ' + nome + '_Units_' + str(units) + '_Dropouts_' + str(dropout) + '_Batchs_' + str(batch_size) + '.txt', 'a+') as f:
        f.write(acc + '\n')
        f.close()
##
## Run, run little Forest!
##
myNet(SEQUENCE_MAXLEN,emb,'LSTM','LSTM',128,0,32,50,x_train,y_train,x_val,y_val,x_test,y_test)
myNet(SEQUENCE_MAXLEN,emb,'LSTM','LSTM',128,0.25,32,50,x_train,y_train,x_val,y_val,x_test,y_test)
myNet(SEQUENCE_MAXLEN,emb,'LSTM','LSTM',128,0.5,32,50,x_train,y_train,x_val,y_val,x_test,y_test)
myNet(SEQUENCE_MAXLEN,emb,'Bidirectional','Bidirectional',32,0,32,50,x_train,y_train,x_val,y_val,x_test,y_test)
myNet(SEQUENCE_MAXLEN,emb,'Bidirectional','Bidirectional',32,0.25,32,50,x_train,y_train,x_val,y_val,x_test,y_test)
myNet(SEQUENCE_MAXLEN,emb,'Bidirectional','Bidirectional',32,0.5,32,50,x_train,y_train,x_val,y_val,x_test,y_test)

exit()

##
## Run, big Forest, run!
##
for net in ['LSTM','Bidirectional']:    
    for units in [32,64,128,256]:
        for dropouts in [0,.25,.5]:
            for batch_size in [16,32,64]:
                print('\n#####################################################################\n')
                print('#####################################################################\n')
                print('Running now: '+ net + '_Units_' + str(units) + '_Dropouts_' + str(dropouts) + '_Batchs_' + str(batch_size))
                print('\n#####################################################################\n')
                print('#####################################################################\n')
                myNet(SEQUENCE_MAXLEN,emb,net,net,units,dropouts,batch_size,50,x_train,y_train,x_val,y_val,x_test,y_test)
            
