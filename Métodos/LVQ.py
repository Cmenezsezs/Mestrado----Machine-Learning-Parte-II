import numpy as np

class LVQ():

    def __init__(self):
        self.prototypes = None
        self.proto_train_labels = []

    # Code by: https://gist.github.com/SilverTear1/5594bc93f2685b82ad4c967aff2da644
    def train(self, train_data, train_labels, num_epochs, learning_rate,
        printar=False):
        num_dims = train_data.shape[1]
        train_labels = train_labels.astype(int)
        unique_train_labels = list(set(train_labels))
        num_protos = len(unique_train_labels)

        self.prototypes = np.empty((num_protos, num_dims))

        for i in unique_train_labels:
            class_train_data = train_data[train_labels == i, :]
            mean = np.mean(class_train_data, axis=0)
            self.prototypes[i] = mean
            self.proto_train_labels.append(i)

        # if printar: print(f"Processando epoca 0 do LVQ...", end="\r")
        for epoch in range(0, num_epochs):
            for fvec, lbl in zip(train_data, train_labels):
                # Compute distance from each prototype to this point
                distances = list(np.sum(np.subtract(fvec, p)**2) for p in self.prototypes)
                min_dist_index = distances.index(min(distances))

                # Determine winner prototype.
                winner = self.prototypes[min_dist_index]
                winner_label = self.proto_train_labels[min_dist_index]

                # Push or repel the prototype based on the label.
                if winner_label == lbl: sign = 1
                else:                   sign = -1

                # Update winner prototype
                self.prototypes[min_dist_index] = np.add(
                    self.prototypes[min_dist_index],
                    np.subtract(fvec, winner) * learning_rate * sign)

            if printar: print(f"Treinou época: {epoch}")
        return (self.prototypes, self.proto_train_labels)


    def test(self, test_data=None, test_labels=None):
        """
        Returns: ACURÁCIA: (acerto / len(test_labels))
        """

        # Use validation set to test performance.
        acerto = 0
        for fvec, lbl in zip(test_data, test_labels):
            distances = list(np.sum(np.subtract(fvec, p) ** 2) for p in self.prototypes)
            min_dist_index = distances.index(min(distances))

            # Determine winner prototype label
            winner_label = self.proto_train_labels[min_dist_index]
            if winner_label == lbl: acerto = acerto + 1

        return (acerto / len(test_labels))

    def predict(self, test_data=None):
        vet = []
        for fvec in test_data:
            distances = list(np.sum(np.subtract(fvec, p) ** 2) for p in self.prototypes)
            min_dist_index = distances.index(min(distances))

            # Determine winner prototype label
            vet.append(self.proto_train_labels[min_dist_index])
        return vet

################################################################################