"""
EXECUÇÃO PARA A BASE DE DADOS "PORTO": 
https://www.kaggle.com/c/porto-seguro-safe-driver-prediction/data

>> TRATAMENTO DOS DADOS E USO DE TODOS OS MODELOS
"""

import numpy as np
import pandas as pd
from time import time
import tensorflow as tf
from metodos.LVQ import LVQ
from sklearn import tree, svm
import matplotlib.pyplot as plt
from sklearn.utils import resample
from sklearn.decomposition import PCA
from metodos.rede_main_3 import RedeTop
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error as mse
from sklearn.model_selection import RepeatedStratifiedKFold

print(f"Programa Iniciado!")

mostrar_estudo_base = False

exec_LVQ = False
exec_KNN = False
exec_TREE = False
exec_SVM = False # Demora muitoo!
exec_NN = True

porcent_test = 0.3
exec_under_down_major = False # False é mais lento (base maior)

seed = 42

################################################################################
# Carrega base de dados
df = pd.read_csv('./base_dados/train_porto.csv', index_col=0)

if mostrar_estudo_base:
    nao_reclamou, reclamou = df.target.value_counts()
    print(f"Casos com reclamação: {nao_reclamou}")
    print(f"Casos sem reclamação: {reclamou}")
    print(f"Proporção sem reclamação: {round(100*nao_reclamou/(df.shape[0]), 2)}%")
    
    print(f"-----")
    print(f"Quantidade de elementos NAN por classe:")
    for col in df.columns:
        count=df[df[col]==-1][col].count()
        if count > 0:
            print(f'\t{col}: \t[{count}] {round(100*count/df.shape[0], 2)}%')

# Remove as colunas que mais possuem falta de valores (NAN)
df = df.drop(["ps_reg_03", "ps_car_03_cat", "ps_car_05_cat"], axis=1)

# Substitui os NAN das colunas que restaram depois do drop
from sklearn.impute import SimpleImputer
cat_cols=['ps_ind_02_cat', 'ps_ind_04_cat', 'ps_ind_05_cat','ps_car_01_cat', 'ps_car_02_cat', 'ps_car_07_cat','ps_car_09_cat']
num_cols=['ps_car_11', 'ps_car_12', 'ps_car_14']
num_imp=SimpleImputer(missing_values=-1,strategy='mean')
cat_imp=SimpleImputer(missing_values=-1,strategy='most_frequent')
for col in cat_cols: df[col]=cat_imp.fit_transform(df[[col]]).ravel()
for col in num_cols: df[col]=num_imp.fit_transform(df[[col]]).ravel()
if mostrar_estudo_base: print(f"Shape da base de dados após corte: {df.shape}")

X_aux = df[df.columns[1:]].to_numpy()
y_aux = df['target'].to_numpy()

escala = MinMaxScaler()
X_aux = escala.fit_transform(X_aux)

# Escolha do PCA
# vet_plot = []
# for n_c in range(2, 15):
#     X_b = X.copy()
#     pca = PCA(n_components=n_c)
#     X = pca.fit_transform(X)
#     X = pca.inverse_transform(X)
#     aux = mse(X_b, X)
#     vet_plot.append(aux)
#     print(n_c, aux)
# plt.plot([x+2 for x in range(len(vet_plot))], vet_plot)
# plt.title("Erro entre base original e base reconstruída após PCA")
# plt.xlabel("Num Componentes")
# plt.ylabel("Erro (MSE)")
# plt.show()
# exit()

# PCA
pca = PCA(n_components=2)
X_aux = pca.fit_transform(X_aux)

atrib_train_aux, atrib_test, labels_train_aux, labels_test = train_test_split(
    X_aux, y_aux, test_size=porcent_test, random_state=seed, shuffle=True, stratify=y_aux)

# https://elitedatascience.com/imbalanced-classes
df_aux = pd.DataFrame(atrib_train_aux)
df_aux['target'] = labels_train_aux # add no fim
major = df_aux[df_aux.target == 0]
minor = df_aux[df_aux.target == 1]
df_minor_upsampled = resample(
    minor, replace=True, n_samples=major.shape[0], random_state=seed)
df_major_downsampled = resample(
    major, replace=False, n_samples=minor.shape[0], random_state=seed)
df_minor_upsampled = pd.concat([major, df_minor_upsampled])
df_major_downsampled = pd.concat([minor, df_major_downsampled])
if mostrar_estudo_base: print(f"df_minor_upsampled:\n{df_minor_upsampled.target.value_counts()}")
if mostrar_estudo_base: print(f"df_major_downsampled:\n{df_major_downsampled.target.value_counts()}")


# for df_base_balan, base_n_aux in zip([df_minor_upsampled, df_major_downsampled], ["up", "down"]):
#     print(f"Executando base {base_n_aux}sampled")
df_base_balan = df_minor_upsampled
if exec_under_down_major: df_base_balan = df_major_downsampled

atrib_train = df_base_balan[df_base_balan.columns[:-1]].to_numpy()
labels_train = df_base_balan['target'].to_numpy()

if mostrar_estudo_base: print(f"len(train): {len(labels_train)}")
if mostrar_estudo_base: print(f"len(test): {len(labels_test)}")

################################################################################
print("-----------------------------------------------------------------------")

lvq = LVQ()
# lr_LVQ = 0.00001 # base major # OK!
lr_LVQ = 0.00001 # base minor
acuracia_vali_LVQ = []

# n_k = 3 # base major # OK!
n_k = 3 # base minor
knn = KNeighborsClassifier(n_neighbors=n_k)
acuracia_vali_KNN = []


# crit = "gini" # base major # OK!
# max_leaf = 30 # base major # OK!
crit = "gini" # base minor
max_leaf = 30 # base minor
tree_model = tree.DecisionTreeClassifier(criterion=crit, max_leaf_nodes=max_leaf)
acuracia_vali_TREE = []

# ["linear", "poly", "rbf", "sigmoid"]
# kernel = "poly" # base major # OK!
kernel = "poly" # base minor
SVM_model = svm.SVC(kernel=kernel)
acuracia_vali_SVM = []

NN_model = RedeTop()
NN_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss=tf.keras.losses.mean_squared_error, metrics=["accuracy"])
acuracia_vali_NN = []

atrib_train_NN, labels_train_NN = [], []
atrib_vali_NN, labels_vali_NN = [], []
rskf = RepeatedStratifiedKFold(n_splits=2, n_repeats=5, random_state=seed)
for count_exec, (index_treino, index_teste) in enumerate(rskf.split(atrib_train, labels_train)):

    if exec_LVQ:
        print(f"Executando LVQ [{count_exec}]")
        lvq.train(atrib_train[index_treino], labels_train[index_treino], 1, lr_LVQ, False)
        acuracia_vali_LVQ.append(lvq.test(atrib_train[index_teste], labels_train[index_teste]))
    if exec_KNN:
        print(f"Executando KNN [{count_exec}]")
        knn.fit(atrib_train[index_treino], labels_train[index_treino])
        acuracia_vali_KNN.append(knn.score(atrib_train[index_teste], labels_train[index_teste]))
    if exec_TREE:
        print(f"Executando TREE [{count_exec}]")
        tree_model = tree_model.fit(atrib_train[index_treino].tolist(), labels_train[index_treino].tolist())
        acuracia_vali_aux = tree_model.predict(atrib_train[index_teste]) == labels_train[index_teste]
        acuracia_vali_TREE.append(sum(acuracia_vali_aux)/len(acuracia_vali_aux))
    if exec_SVM:
        print(f"Executando SVM [{count_exec}]")
        # pega pequena porcentagem do fold de treino e vali por conta do tempo
        atrib_train_SVM, _, labels_train_SVM, _ = train_test_split(
            atrib_train[index_treino], labels_train[index_treino], test_size=0.95,
            random_state=seed, shuffle=True, stratify=labels_train[index_treino])
        atrib_vali_SVM, _, labels_vali_SVM, _ = train_test_split(
            atrib_train[index_teste], labels_train[index_teste], test_size=0.9,
            random_state=seed, shuffle=True, stratify=labels_train[index_teste])

        SVM_model.fit(atrib_train_SVM, labels_train_SVM)
        acuracia_vali_aux = SVM_model.predict(atrib_vali_SVM) == labels_vali_SVM
        acuracia_vali_SVM.append(sum(acuracia_vali_aux)/len(acuracia_vali_aux))
    if exec_NN:
        print(f"Executando NN [{count_exec}]")
        for asd in atrib_train[index_treino]:
            atrib_train_NN.append(list(asd))
        for asd in labels_train[index_treino]:
            labels_train_NN.append(asd)

        for asd in atrib_train[index_teste]:
            atrib_vali_NN.append(list(asd))
        for asd in labels_train[index_teste]:
            labels_vali_NN.append(asd)

# Executa o fit apenas uma vez
# (https://github.com/keras-team/keras/issues/4446)
if exec_NN:
    atrib_train_NN, labels_train_NN = np.asarray(atrib_train_NN), np.asarray(labels_train_NN)
    atrib_vali_NN, labels_vali_NN = np.asarray(atrib_vali_NN), np.asarray(labels_vali_NN)
    NN_model.fit(atrib_train_NN, labels_train_NN, batch_size=32, epochs=2)
    acuracia_vali_NN.append(NN_model.evaluate(atrib_vali_NN, labels_vali_NN)[1])



if exec_LVQ:
    acu_test_LVQ = lvq.test(atrib_test, labels_test)
    pred_LVQ_test = lvq.predict(atrib_test)
    print(f"Média Acurácia VALI LVQ: {sum(acuracia_vali_LVQ)/len(acuracia_vali_LVQ)*100:.3f}%")
    print(f"Média Acurácia TEST LVQ: {acu_test_LVQ*100:.3f}%")
    print(confusion_matrix(pred_LVQ_test, labels_test))
if exec_KNN:
    acu_test_KNN = knn.score(atrib_test, labels_test)
    pred_KNN_test = knn.predict(atrib_test)
    print(f"Média Acurácia VALI KNN: {sum(acuracia_vali_KNN)/len(acuracia_vali_KNN)*100:.3f}%")
    print(f"Média Acurácia TEST KNN: {acu_test_KNN*100:.3f}%")
    print(confusion_matrix(pred_KNN_test, labels_test))
if exec_TREE:
    pred_TREE_test = tree_model.predict(atrib_test.tolist())
    acuracia_test_TREE = ((pred_TREE_test == labels_test).sum())/len(labels_test)
    print(f"Média Acurácia VALI TREE: {sum(acuracia_vali_TREE)/len(acuracia_vali_TREE)*100:.3f}%")
    print(f"Média Acurácia TEST TREE: {acuracia_test_TREE*100:.3f}%")
    print(confusion_matrix(pred_TREE_test, labels_test))
if exec_SVM:
    pred_SVM_test = SVM_model.predict(atrib_test)
    acuracia_test_SVM = ((pred_SVM_test == labels_test).sum())/len(labels_test)
    print(f"Média Acurácia VALI SVM: {sum(acuracia_vali_SVM)/len(acuracia_vali_SVM)*100:.3f}%")
    print(f"Média Acurácia TEST SVM: {acuracia_test_SVM*100:.3f}%")
    print(confusion_matrix(pred_SVM_test, labels_test))
if exec_NN:
    pred_NN_test = NN_model.evaluate(atrib_test, labels_test)[1]
    print(f"Média Acurácia VALI NN: {sum(acuracia_vali_NN)/len(acuracia_vali_NN)*100:.3f}%")
    print(f"Média Acurácia TEST NN: {pred_NN_test*100:.3f}%")
    pred_test = NN_model.predict(atrib_test)
    pred_test[pred_test >= 0.5] = 1; pred_test[pred_test < 0.5] = 0;
    print(confusion_matrix(pred_test, labels_test))