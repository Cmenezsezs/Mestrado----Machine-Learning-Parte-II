from time import time
import numpy as np
from base_dados.BalancedScale import get_dataset
from metodos.LVQ import LVQ
from sklearn.neighbors import KNeighborsClassifier
from sklearn import tree
from sklearn import svm
from sklearn.model_selection import RepeatedKFold
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score
import matplotlib.pyplot as plt
import graphviz 

exec_LVQ = False
exec_KNN = False
exec_TREE = False
exec_SVM = False


################################################################################
# Carrega base de dados

dir_database = "./base_dados/balance-scale.data"
# seeds: 24, 47, 33, 29, 47, 48, 49, 6, 19
atrib_train, labels_train, atrib_test, labels_test, labels_names = get_dataset(
    dir_database, proporcao_train_dataset=0.8, printar=False, seed=49)
feature_names = ["Left-Weight", "Left-Distance", "Right-Weight", "Right-Distance"]

################################################################################
print("-----------------------------------------------------------------------")
################################################################################

if exec_LVQ:
    learning_rate_LVQ = 0.1
    num_epochs_LVQ = 2
    lvq = LVQ()
    t_lvq_train = time()
    lvq.train(atrib_train, labels_train, num_epochs_LVQ, learning_rate_LVQ, False)
    t_lvq_train = time() - t_lvq_train; t_lvq_teste = time()
    acuracia_LVQ = lvq.test(atrib_test, labels_test)
    t_lvq_teste = time() - t_lvq_teste
    print(f"Acurácia LVQ: {acuracia_LVQ*100:.5f}%")
    print(f"Tempo de execução do TREINAMENTO do LVQ: {t_lvq_train:.5f} seg.")
    print(f"Tempo de execução do TESTE do LVQ: {t_lvq_teste:.5f} seg.")
    print(f"Tempo de execução TOTAL do LVQ: {(t_lvq_train+t_lvq_teste):.5f} seg.")

    ############################################################################
    print("-------------------------------------------------------------------")
    ############################################################################

if exec_KNN:
    knn = KNeighborsClassifier(n_neighbors=len(list(set(labels_test))))
    t_knn_total = time()
    knn.fit(atrib_train, labels_train)
    print(f"Acurácia KNN: {knn.score(atrib_test, labels_test)*100:.5f}%")
    print(f"Tempo de execução do KNN: {(time()-t_knn_total):.5f} seg.")

    ############################################################################
    print("-------------------------------------------------------------------")
    ############################################################################

if exec_TREE:
    qnt_testes_folha = 35
    for crit in ["entropy", "gini"]:
        x_plt = [x for x in range(3, qnt_testes_folha)]
        y_test_plt, y_train_plt = [], []
        for c in range(3, qnt_testes_folha):
            print(f"Max Folhas: {c}, Criterion: {crit}")
            clf = tree.DecisionTreeClassifier(criterion=crit, max_leaf_nodes=c)
            t_arvore = time()
            clf = clf.fit(atrib_train, labels_train)

            pred_test = clf.predict(atrib_test)
            t_arvore = time()-t_arvore
            print(f"Tempo de execução da Árvore de Decisão: {t_arvore:.5f} seg.")
            acuracia_test = ((pred_test == labels_test).sum())/len(labels_test)
            print(f"Acurácia TEST Arv. Decisão: {acuracia_test*100:.5f}%")
            y_test_plt.append(acuracia_test*100)

            ### Coleta acurácia da base de treino para verificar overfiting
            pred_train = clf.predict(atrib_train)
            acuracia = (pred_train == labels_train).sum()/len(labels_train)
            print(f"Acurácia TRAIN Arv. Decisão: {acuracia*100:.5f}%")
            y_train_plt.append(acuracia*100)
            print("---------")

        # Constrói a Árvore em PDF
        print(f"feature_names: {feature_names}"); print(f"labels_names: {labels_names}")
        # dot_data = tree.export_graphviz(clf, out_file=None, feature_names=feature_names,
        #     class_names=[x[1] for x in labels_names], filled=True, rounded=True,
        #     special_characters=True)
        # graph = graphviz.Source(dot_data)  
        # graph.render("arvore")
        
        plt.plot(x_plt, y_train_plt, '-b', label="train dataset")
        plt.plot(x_plt, y_test_plt, '-r', label="test dataset")
        plt.xlabel('qnt max folhas')
        plt.ylabel('acurácia')
        plt.title(f"Acurácia da Árvore de Decisão ({crit})")
        plt.legend(); plt.show()



    ############################################################################
    print("-------------------------------------------------------------------")
    ############################################################################


if exec_SVM:

    num_splits = 5
    num_repeats = 10
    rkf = RepeatedKFold(n_splits=num_splits, n_repeats=num_repeats, random_state=49)
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
        
        print(f"Acurácia [VALI] do kernel {k}:", end="  \t")
        print(f"{media_kernel/(num_splits*num_repeats):.3f}%")

        print(f"Acurácia [TESTE] do kernel {k}:", end=" \t")
        predicao = clf.predict(atrib_test)
        print(f"{(np.sum(labels_test == predicao)/len(labels_test))*100:.3f}%")
        print(confusion_matrix(predicao, labels_test))
        print(f"Tempo de Execução: {time()-t1:.5f}s")
        preci = precision_score(predicao, labels_test, average=None)*100
        print(f"Precisão: {list(zip(preci, [x[1] for x in labels_names]))}")
        print(f"----")


    ############################################################################
    print("-------------------------------------------------------------------")
    ############################################################################