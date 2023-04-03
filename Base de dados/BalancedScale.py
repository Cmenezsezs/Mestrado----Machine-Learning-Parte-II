import numpy as np
import random
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder

# https://archive.ics.uci.edu/ml/datasets/balance+scale

def get_dataset(dir_database, proporcao_train_dataset, seed=None, printar=False,):
    """ Returns: atrib_train, labels_train, atrib_test, labels_test """

    if seed != None: random.seed(seed)

    atrib = []; labels = []
    arquivo = open(dir_database, "r")
    linhas = arquivo.readlines()[:-1]; random.shuffle(linhas)
    for i in linhas:
        linha = i.split(',')
        if len(linha) == 0: continue
        labels.append(linha[0])
        atrib.append([int(x) for x in linha[1:]])
    arquivo.close()
    labels_bkp = labels

    # Normaliza atributos entre 0 e 1 e cria um encode para os labels
    atrib = MinMaxScaler(feature_range=(0, 1)).fit_transform(atrib) # TODO:
    labels = LabelEncoder().fit_transform(labels)
    num_samples = len(atrib)

    atrib_train = atrib[:int((num_samples-1)*proporcao_train_dataset)]
    labels_train = labels[:int((num_samples-1)*proporcao_train_dataset)]
    atrib_test = atrib[int((num_samples-1)*proporcao_train_dataset):]
    labels_test = labels[int((num_samples-1)*proporcao_train_dataset):]

    if printar:
        print(f"Type: {type(atrib), type(labels)}")
        print(f"len(atrib): {len(atrib)}, len(labels): {len(labels)}")
        print(f"Integridade: {not False in [len(x) == len(atrib[0]) for x in atrib]}")

    labels_names = list(set(zip(list(labels), list(labels_bkp))))
    labels_names = sorted(labels_names, key=lambda x: x[0])
    return atrib_train, labels_train, atrib_test, labels_test, labels_names