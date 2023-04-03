import numpy as np
from time import time, sleep
from random import shuffle
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.neighbors import KNeighborsClassifier

################################################################################
# Carrega base de dados

proporcao_train_dataset = 0.8

################################################################################
# Executa K-NN
print("\n")

knn = KNeighborsClassifier(n_neighbors=num_labels)
t_knn_total = time()
knn.fit(atrib_train, labels_train)
print(f"Acurácia KNN: {knn.score(atrib_test, labels_test)*100:.5f}%")
print(f"Tempo de execução do KNN: {(time()-t_knn_total):.5f} seg.")

################################################################################
# Executa LVQ
print("\n")

# Code by: https://gist.github.com/SilverTear1/5594bc93f2685b82ad4c967aff2da644
def train_test_lvq(train_data, train_labels, num_epochs, learning_rate,
    test_data=None, test_labels=None):
    t_lvq_total = time()

    num_dims = train_data.shape[1]
    train_labels = train_labels.astype(int)
    unique_train_labels = list(set(train_labels))

    num_protos = len(unique_train_labels)
    prototypes = np.empty((num_protos, num_dims))
    proto_train_labels = []

    t_lvq_train = time()
    for i in unique_train_labels:
        class_train_data = train_data[train_labels == i, :]
        mean = np.mean(class_train_data, axis=0)
        prototypes[i] = mean
        proto_train_labels.append(i)

    for epoch in range(0, num_epochs):
        for fvec, lbl in zip(train_data, train_labels):
            # Compute distance from each prototype to this point
            distances = list(np.sum(np.subtract(fvec, p)**2) for p in prototypes)
            min_dist_index = distances.index(min(distances))

            # Determine winner prototype.
            winner = prototypes[min_dist_index]
            winner_label = proto_train_labels[min_dist_index]

            # Push or repel the prototype based on the label.
            if winner_label == lbl: sign = 1
            else:                   sign = -1

            # Update winner prototype
            prototypes[min_dist_index] = np.add(
                prototypes[min_dist_index], np.subtract(fvec, winner) * learning_rate * sign)

        print(f"Treinou época: {epoch}")
    
    print(f"Tempo de execução do TREINAMENTO do LVQ: {(time()-t_lvq_train):.5f} seg.")


    t_lvq_teste = time()
    # Use validation set to test performance.
    acerto = 0
    for fvec, lbl in zip(test_data, test_labels):
        distances = list(np.sum(np.subtract(fvec, p) ** 2) for p in prototypes)
        min_dist_index = distances.index(min(distances))

        # Determine winner prototype label
        winner_label = proto_train_labels[min_dist_index]
        if winner_label == lbl: acerto = acerto + 1

    print(f"Tempo de execução do TESTE do LVQ: {(time()-t_lvq_teste):.5f} seg.")
    print(f"Tempo de execução TOTAL do LVQ: {(time()-t_lvq_total):.5f} seg.")
    print(f"Acurácia LVQ: {acerto / len(test_labels)*100:.5f}%")
    return (prototypes, proto_train_labels)

lr_LVQ = 0.1
num_epochs = 2

train_test_lvq(train_data=atrib_train, train_labels=labels_train,
    num_epochs=num_epochs, learning_rate=lr_LVQ, test_data=atrib_test,
    test_labels=labels_test)


################################################################################