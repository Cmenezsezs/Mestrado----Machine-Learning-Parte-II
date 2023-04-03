"""
Esse script faz uso da base de dados GLASS, e é utilizado para averiguar a
    execução do modelo SVM (com busca de kernels) e Rede Neural (com busca de
    hiperparâmetros)
"""


import optuna
import plotly
import random
import numpy as np
from time import time
from sklearn import svm
import tensorflow as tf
from base_dados.Glass import get_dataset
from sklearn.model_selection import RepeatedKFold
from sklearn.metrics import precision_score, confusion_matrix


exec_SVM = False
exec_rede = True
seed = 49

random.seed(seed)
np.random.seed(seed)
tf.random.set_seed(seed)


################################################################################
# Carrega base de dados

dir_database = "./base_dados/glass.data"
# seeds: 24, 47, 33, 29, 47, 48, 49, 6, 19
atrib_train, labels_train, atrib_test, labels_test = get_dataset(
# atrib_train, labels_train, atrib_test, labels_test, labels_names = get_dataset(
    dir_database, proporcao_train_dataset=0.7,
    printar=False, seed=seed, normalizar=True)
feature_names = ["RI", "Na", "Mg", "Al", "Si", "K", "Ca", "Ba", "Fe"]

################################################################################
print("-----------------------------------------------------------------------")
################################################################################

if exec_SVM:

    num_splits = 5
    num_repeats = 10
    rkf = RepeatedKFold(n_splits=num_splits, n_repeats=num_repeats, random_state=seed)
    kernels = ["linear", "poly", "rbf", "sigmoid"] # "precomputed"
    for k in kernels:
        t1 = time()
        media_kernel = 0.0
        for train_index, vali_index in rkf.split(atrib_train):
            X_train, X_vali = atrib_train[train_index], atrib_train[vali_index]
            y_train, y_vali = labels_train[train_index], labels_train[vali_index]

            clf = svm.SVC(kernel=k)
            clf.fit(X_train, y_train)

            predicao = clf.predict(X_vali)
            media_kernel += (np.sum(y_vali == predicao)/len(y_vali))*100
        
        # print(f"Acurácia [VALI] do kernel {k}:", end="  \t")
        # print(f"{media_kernel/(num_splits*num_repeats):.3f}%")
        #
        print(f"Acurácia [Treino] do kernel {k}:", end=" \t")
        predicao = clf.predict(atrib_train)
        print(f"{(np.sum(labels_train == predicao)/len(labels_train))*100:.3f}%")
        #
        print(f"Acurácia [TESTE] do kernel {k}:", end=" \t")
        predicao = clf.predict(atrib_test)
        print(f"{(np.sum(labels_test == predicao)/len(labels_test))*100:.3f}%")
        #
        print(confusion_matrix(predicao, labels_test))
        print(f"Tempo de Execução: {time()-t1:.5f}s")
        preci = precision_score(predicao, labels_test, average=None)*100
        # print(f"Precisão: {list(zip(preci, [x[1] for x in labels_names]))}")
        print(f"----")

    ############################################################################
    print("-------------------------------------------------------------------")
    ############################################################################





# Criando modelo
def neural_network(trial):
    n_layers = trial.suggest_int('n_layers', 1, 3)
    weight_decay = trial.suggest_float("weight_decay", 1e-10, 1e-3, log=True)
    activation_options = ["relu", "sigmoid", "linear", "tanh", "softmax", "exponential"]
    activation_selection = trial.suggest_categorical("activation", activation_options)
    breakpoint()
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Flatten())
    for i in range(n_layers):
        num_hidden = trial.suggest_int("n_units_l{}".format(i), 4, 128, log=True)
        model.add(
            tf.keras.layers.Dense(
                num_hidden,
                activation=activation_selection,
            )
        )
    model.add(
        tf.keras.layers.Dense(7, "softmax")
    )
    return model


def create_optimizer(trial):
    # We optimize the choice of optimizers as well as their parameters.
    kwargs = {}
    optimizer_options = ["Adam"]
    # optimizer_options = ["RMSprop", "Adam", "SGD"]
    optimizer_selected = trial.suggest_categorical("optimizer", optimizer_options)
    learning_rate_selected = trial.suggest_float("learning_rate", 1e-3, 1e-2, log=True)
    if optimizer_selected == "RMSprop":
        optimizer=tf.keras.optimizers.RMSprop(learning_rate=learning_rate_selected)
    elif optimizer_selected == "Adam":
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate_selected)
    elif optimizer_selected == "SGD":
        optimizer=tf.keras.optimizers.SGD(learning_rate=learning_rate_selected)
    return optimizer


def objective(trial):
    # 1. Abrindo Dataset
    X_train, y_train, X_test, y_test = atrib_train, labels_train, atrib_test, labels_test

    # 2. Suggest values of the hyperparameters using a trial object.
    n_layers = trial.suggest_int('n_layers', 4, 4)
    # activation_options = ["relu", "sigmoid", "tanh", "softmax"]
    activation_options = ["tanh"]
    activation_selection = trial.suggest_categorical("activation", activation_options)
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Flatten())
    arquitetura = []
    for i in range(n_layers):
        num_hidden = trial.suggest_int(f'n_units_l{i}', 128, 256, log=True)
        arquitetura.append(num_hidden)
        model.add(tf.keras.layers.Dense(num_hidden, activation=activation_selection))
    model.add(tf.keras.layers.Dense(6, "softmax"))
    otimizador = create_optimizer(trial)
    model.compile(loss='sparse_categorical_crossentropy',optimizer=otimizador, metrics=['accuracy'])
    log_dir = "logs/fit/" + otimizador._name + "_hidlay-"+str(n_layers)+ "_arq" + str(arquitetura)+"_ativacao-"+activation_selection+"_lr-"+str(np.array(otimizador.lr))
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
    

    num_splits = 5
    num_repeats = 10
    rkf = RepeatedKFold(n_splits=num_splits, n_repeats=num_repeats, random_state=seed)
    for train_index, vali_index in rkf.split(atrib_train):
        X_train, X_vali = atrib_train[train_index], atrib_train[vali_index]
        y_train, y_vali = labels_train[train_index], labels_train[vali_index]
        model.fit(
            X_train, y_train, epochs=10, verbose=0,
            callbacks=[tensorboard_callback], validation_data=(X_vali, y_vali))
    
    saida = model.evaluate(X_test, y_test)
    return saida[1]
 

# Procurar hiperparametros
procurar_hiper = False

if exec_rede:
    if procurar_hiper:
        # 3. Create a study object and optimize the objective function.
        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=10)

        print("Number of finished trials: ", len(study.trials))

        print("Best trial:")
        trial = study.best_trial

        print("  Value: ", trial.value)

        print("  Params: ")
        for key, value in trial.params.items():
            print("    {}: {}".format(key, value))
    else:
        X_train, y_train, X_test, y_test = atrib_train, labels_train, atrib_test, labels_test

        model = tf.keras.Sequential()
        model.add(tf.keras.layers.Flatten())
        model.add(tf.keras.layers.Dense(128, "tanh"))
        # model.add(tf.keras.layers.Dense(256, "tanh"))
        model.add(tf.keras.layers.Dense(256, "tanh"))
        model.add(tf.keras.layers.Dense(128, "tanh"))
        model.add(tf.keras.layers.Dense(6, "softmax"))
        model.compile(
            loss='sparse_categorical_crossentropy',
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.00155),
            metrics=['accuracy'])


        num_splits = 5
        num_repeats = 10
        rkf = RepeatedKFold(n_splits=num_splits, n_repeats=num_repeats, random_state=seed)
        sum_vali = 0.0
        for train_index, vali_index in rkf.split(atrib_train):
            X_train, X_vali = atrib_train[train_index], atrib_train[vali_index]
            y_train, y_vali = labels_train[train_index], labels_train[vali_index]
            model.fit(
                X_train, y_train, epochs=5, verbose=0,
                validation_data=(X_vali, y_vali))


        acuracia_treino = model.evaluate(atrib_train, labels_train)[1]
        print(f"Acurácia da Treinamento: {acuracia_treino}")

        acuracia_test = model.evaluate(atrib_test, labels_test)[1]
        print(f"Acurácia da Teste: {acuracia_test}")

        print("\n\n")
        
        # Essa predição é a probabilidade de cada classe (classe = indice)
        pred = model.predict(atrib_test)
        
        # Pega os indices/classes das maiores probabilidades da predição
        indices = []
        for xi in range(len(pred)):
            indices.append(pred[xi].argmax())
        indices = np.asarray(indices)
        print(indices)

        print(confusion_matrix(indices, labels_test))
