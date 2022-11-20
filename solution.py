import numpy as np
# import matplotlib.pyplot as plt


class SVM:
    def __init__(self, eta, C, niter, batch_size, verbose):
        self.eta = eta
        self.C = C
        self.niter = niter
        self.batch_size = batch_size
        self.verbose = verbose

    def make_one_versus_all_labels(self, y, m):
        """
        y : numpy array of shape (n,)
        m : int (num_classes)
        returns : numpy array of shape (n, m)
        """
        n = y.shape[0]

        labels = np.full((n, m), -1)

        for index, label in enumerate(y):
            labels[index, label] = 1

        return labels

    def compute_loss(self, x, y):
        """
        x : numpy array of shape (minibatch size, num_features)
        y : numpy array of shape (minibatch size, num_classes)
        returns : float
        """
        total_loss = 0

        n = x.shape[0]
        m = y.shape[1]

        # itération sur les classes
        for j in range(m):

            # itération sur les exemples (xi, yi)
            for i in range(n):

                total_loss += np.power(np.maximum(0.0, 2 - np.dot(np.transpose(self.w)[j], x[i]) * y[i][j]), 2) / n

        # ajouter la régularisation
        total_loss += (self.C / 2) * (self.w ** 2).sum()

        return total_loss

    def compute_gradient(self, x, y):
        """
        x : numpy array of shape (minibatch size, num_features)
        y : numpy array of shape (minibatch size, num_classes)
        returns : numpy array of shape (num_features, num_classes)
        """
        n = x.shape[0]

        # shape (n, m) = (5882, 6)
        predictions = np.dot(x, self.w)

        # shape (n, m) = (5882, 6)
        activation = 2 * np.maximum(0.0, (2 - y * predictions))

        # shape (f, m) (562, 6)
        regularisation = self.C * self.w

        gradient = -(np.dot(x.T, activation * y)) / n

        return gradient + regularisation

    # Batcher function
    def minibatch(self, iterable1, iterable2, size=1):
        l = len(iterable1)
        n = size
        for ndx in range(0, l, n):
            index2 = min(ndx + n, l)
            yield iterable1[ndx: index2], iterable2[ndx: index2]

    def infer(self, x):
        """
        x : numpy array of shape (num_examples_to_infer, num_features)
        returns : numpy array of shape (num_examples_to_infer, num_classes)
        """
        n = x.shape[0]
        m = self.w.shape[1]

        y_inferred = np.full((n, m), -1)

        # itération sur les x
        for i in range(n):

            loss_by_class = []

            # itération sur les classes
            for j in range(m):

                y = np.full(6, -1)
                y[j] = 1
                loss = self.compute_loss(x[i][np.newaxis, :], y[np.newaxis, :])
                loss_by_class.append(loss)

            class_index = np.argmin(loss_by_class)

            y_inferred[i][class_index] = 1

        return y_inferred

    def compute_accuracy(self, y_inferred, y):
        """
        y_inferred : numpy array of shape (num_examples, num_classes)
        y : numpy array of shape (num_examples, num_classes)
        returns : float
        """
        n_accurate = 0
        n_test = len(y_inferred)
        for i in range(n_test):
            if np.array_equal(y_inferred[i], y[i]):
                n_accurate += 1

        accuracy = n_accurate / n_test
        return accuracy

    def fit(self, x_train, y_train, x_test, y_test):
        """
        x_train : numpy array of shape (number of training examples, num_features)
        y_train : numpy array of shape (number of training examples, num_classes)
        x_test : numpy array of shape (number of training examples, nujm_features)
        y_test : numpy array of shape (number of training examples, num_classes)
        returns : float, float, float, float
        """
        self.num_features = x_train.shape[1]
        self.m = y_train.max() + 1
        y_train = self.make_one_versus_all_labels(y_train, self.m)
        y_test = self.make_one_versus_all_labels(y_test, self.m)
        self.w = np.zeros([self.num_features, self.m])

        train_losses = []
        train_accs = []
        test_losses = []
        test_accs = []

        for iteration in range(self.niter):
            # Train one pass through the training set
            for x, y in self.minibatch(x_train, y_train, size=self.batch_size):
                grad = self.compute_gradient(x, y)
                self.w -= self.eta * grad

            # Measure loss and accuracy on training set
            train_loss = self.compute_loss(x_train, y_train)
            y_inferred = self.infer(x_train)
            train_accuracy = self.compute_accuracy(y_inferred, y_train)

            # Measure loss and accuracy on test set
            test_loss = self.compute_loss(x_test, y_test)
            y_inferred = self.infer(x_test)
            test_accuracy = self.compute_accuracy(y_inferred, y_test)

            if self.verbose:
                print(f"Iteration {iteration} | Train loss {train_loss:.04f} | Train acc {train_accuracy:.04f} |"
                      f" Test loss {test_loss:.04f} | Test acc {test_accuracy:.04f}")

            # Record losses, accs
            train_losses.append(train_loss)
            train_accs.append(train_accuracy)
            test_losses.append(test_loss)
            test_accs.append(test_accuracy)

        return train_losses, train_accs, test_losses, test_accs


# DO NOT MODIFY THIS FUNCTION
# Data should be downloaded from the below url, and the
# unzipped folder should be placed in the same directory
# as your solution file:.
# https://drive.google.com/file/d/0Bz9_0VdXvv9bX0MzUEhVdmpCc3c/view?usp=sharing&resourcekey=0-BirYbvtYO-hSEt09wpEBRw
def load_data():
    # Load the data files
    print("Loading data...")
    data_path = "Smartphone Sensor Data/train/"
    x = np.genfromtxt(data_path + "X_train.txt")
    y = np.genfromtxt(data_path + "y_train.txt", dtype=np.int64) - 1
    
    # Create the train/test split
    x_train = np.concatenate([x[0::5], x[1::5], x[2::5], x[3::5]], axis=0)
    x_test = x[4::5]
    y_train = np.concatenate([y[0::5], y[1::5], y[2::5], y[3::5]], axis=0)
    y_test = y[4::5]

    # normalize the data
    mean = x_train.mean(axis=0)
    std = x_train.std(axis=0)
    x_train = (x_train - mean) / std
    x_test = (x_test - mean) / std

    # add implicit bias in the feature
    x_train = np.concatenate([x_train, np.ones((x_train.shape[0], 1))], axis=1)
    x_test = np.concatenate([x_test, np.ones((x_test.shape[0], 1))], axis=1)

    return x_train, y_train, x_test, y_test


# if __name__ == "__main__":
#
#     x_train, y_train, x_test, y_test = load_data()
#
#     print("Fitting the model...")
#     svm = SVM(eta=0.0001, C=2, niter=200, batch_size=100, verbose=False)
#     train_losses, train_accs, test_losses, test_accs = svm.fit(x_train, y_train, x_test, y_test)
#
#     # # to infer after training, do the following:
#     # y_inferred = svm.infer(x_test)
#
#     ## to compute the gradient or loss before training, do the following:
#     # y_train_ova = svm.make_one_versus_all_labels(y_train, 6) # one-versus-all labels
#     # svm.w = np.zeros([x_train.shape[1], 6])
#     # grad = svm.compute_gradient(x_train, y_train_ova)
#     # loss = svm.compute_loss(x_train, y_train_ova)

if __name__ == "__main__":
    x_train, y_train, x_test, y_test = load_data()

    CS = [1, 5, 10]
    niter = 3   # FIXME : Utiliser 200 quand les graphs seront bien mis en forme

    results = []

    for C in CS:
        print("Fitting the model... with C = {}".format(C))
        svm = SVM(eta=0.0001, C=C, niter=niter, batch_size=100, verbose=False)
        # train_losses, train_accs, test_losses, test_accs = svm.fit(x_train, y_train, x_test, y_test)
        result = svm.fit(x_train, y_train, x_test, y_test)
        results.append(result)

    # fig, axs = plt.subplots(2, 2)
    #
    # epoch = range(niter)
    #
    # a1 = axs[0, 0].plot(epoch, results[0][0], 'tab:green')
    # a2 = axs[0, 0].plot(epoch, results[1][0], 'tab:orange')
    # a3 = axs[0, 0].plot(epoch, results[2][0], 'tab:red')
    # axs[0, 0].set_title('Train Loss')
    #
    #
    # axs[0, 1].plot(epoch, results[0][1], 'tab:green')
    # axs[0, 1].plot(epoch, results[1][1], 'tab:orange')
    # axs[0, 1].plot(epoch, results[2][1], 'tab:red')
    # axs[0, 1].set_title('Train Accuracy')
    #
    # axs[1, 0].plot(epoch, results[0][2], 'tab:green')
    # axs[1, 0].plot(epoch, results[1][2], 'tab:orange')
    # axs[1, 0].plot(epoch, results[2][2], 'tab:red')
    # axs[1, 0].set_title('Test Loss')
    #
    # axs[1, 1].plot(epoch, results[0][3], 'tab:green')
    # axs[1, 1].plot(epoch, results[1][3], 'tab:orange')
    # axs[1, 1].plot(epoch, results[2][3], 'tab:red')
    # axs[1, 1].set_title('Test Accuracy')
    #
    # # axs[0][1].set_xticklabels(['a', 'b', 'c', 'd'])
    #
    # fig.legend([a1, a2, a3], labels=["C=1", "C=5", "C=10"], loc='upper right')
    #
    # plt.subplots_adjust(left=0.1,
    #                     bottom=0.1,
    #                     right=0.9,
    #                     top=0.9,
    #                     wspace=0.4,
    #                     hspace=0.4)
    #
    # plt.show()


# if __name__ == "__main__":
#
#     x_train, y_train, x_test, y_test = load_data()
#
#     print("Fitting the model...")
#     svm = SVM(eta=0.0001, C=2, niter=1, batch_size=100, verbose=False)
#     train_losses, train_accs, test_losses, test_accs = svm.fit(x_train, y_train, x_test, y_test)
#
#     # # to infer after training, do the following:
#     # y_inferred = svm.infer(x_test)
#
#     # ## to compute the gradient or loss before training, do the following:
#     y_train_ova = svm.make_one_versus_all_labels(y_train, 6) # one-versus-all labels
#     svm.w = np.zeros([x_train.shape[1], 6])
#     #
#     loss = svm.compute_loss(x_train, y_train_ova)
#     print(loss)
#     #
#     grad = svm.compute_gradient(x_train, y_train_ova)
#     print(grad)
