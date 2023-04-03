import random
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split



# https://archive.ics.uci.edu/ml/machine-learning-databases/glass/

def get_dataset(dir_database, proporcao_train_dataset, seed=None, printar=False,
    normalizar=True):
    """ Returns: atrib_train, labels_train, atrib_test, labels_test """

    arquivo = pd.read_csv(dir_database, sep=",", header=None, index_col=0)
    atributos = ("RI", "Na", "Mg", "Al", "Si", "K", "Ca", "Ba", "Fe", "Vridru")
    arquivo.columns = atributos

    # Particiona entre treino e teste de forma estratificada
    atrib_train, atrib_test, labels_train, labels_test, = train_test_split(
        arquivo[arquivo.columns[:-1]], # Tudo menos os labels/classes
        arquivo[arquivo.columns[-1]], # Labels/Classes
        test_size=1-proporcao_train_dataset,
        stratify=arquivo[arquivo.columns[-1]] # Estratifica com base nos labels
    )

    # Para trabalhar com numpy array
    atrib_train = np.asarray(atrib_train.values.tolist())
    labels_train = np.asarray(labels_train.values.tolist())
    atrib_test = np.asarray(atrib_test.values.tolist())
    labels_test = np.asarray(labels_test.values.tolist())

    # Aplica o Label Encoder
    labels = LabelEncoder().fit_transform(np.append(labels_train, labels_test))
    labels_train = labels[:len(labels_train)]
    labels_test = labels[len(labels_train):]

    # Normaliza entre 0 e 1
    if normalizar:
        normalizer = MinMaxScaler()
        atrib_train = normalizer.fit_transform(atrib_train)
        atrib_test = normalizer.transform(atrib_test)

    return atrib_train, labels_train, atrib_test, labels_test 




if __name__ == '__main__':

    seed = 1
    dir_database = "./glass.data"
    # seeds: 24, 47, 33, 29, 47, 48, 49, 6, 19
    atrib_train, labels_train, atrib_test, labels_test = get_dataset(
        dir_database, proporcao_train_dataset=0.5, printar=False, seed=seed)
    # feature_names = ["RI", "Na", "Mg", "Al", "Si", "K", "Ca", "Ba", "Fe"]





# import pandas as pd
# import plotly.express as px
# import pandas as pd
# atrib = atrib_train
# data = {
#     "RI": atrib[:,0],
#     "Na": atrib[:,1],
#     "Mg": atrib[:,2],
#     "Al": atrib[:,3],
#     "Si": atrib[:,4],
#     "K": atrib[:,5],
#     "Ca": atrib[:,6],
#     "Ba": atrib[:,7],
#     "Fe": atrib[:,8],
# }

# df = pd.DataFrame(data)
# print(df)

# fig = px.line(
#     df,
#     # x=atributos,
#     y=df.columns[0:-1],
#     title="tituloaaaaaa",
#         # labels={
#     #     "Timestamps": "cvbcvb",
#     #     "value": "xxxxx",
#     #     "variable": "asdasd"
#     # }
#     ) 

# fig.show()
# fig.write_html("asd.html")