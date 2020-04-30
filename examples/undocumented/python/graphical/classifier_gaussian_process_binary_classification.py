# This software is distributed under BSD 3-clause license (see LICENSE file).
#
# Authors: Roman Votyakov
import itertools

import matplotlib.pyplot as plt
import numpy as np


def generate_toy_data(n_train=100, mean_a=np.asarray([0, 0]), std_dev_a=1.0, mean_b=3, std_dev_b=0.5):
    # positive examples are distributed normally
    X1 = (np.random.randn(n_train, 2) * std_dev_a + mean_a).T

    # negative examples have a "ring"-like form
    r = np.random.randn(n_train) * std_dev_b + mean_b
    angle = np.random.randn(n_train) * 2 * np.pi
    X2 = np.array([r * np.cos(angle) + mean_a[0], r * np.sin(angle) + mean_a[1]])

    # stack positive and negative examples in a single array
    X_train = np.hstack((X1, X2))

    # label positive examples with +1, negative with -1
    y_train = np.zeros(n_train * 2)
    y_train[:n_train] = 1
    y_train[n_train:] = -1

    return [X_train, y_train]


def gaussian_process_binary_classification_laplace(X_train, y_train, n_test=50):
    import shogun as sg

    # convert training data into Shogun representation
    train_features = sg.create_features(X_train)
    train_labels = sg.create_labels(y_train)

    # generate all pairs in 2d range of testing data
    x1 = np.linspace(X_train[0, :].min() - 1, X_train[0, :].max() + 1, n_test)
    x2 = np.linspace(X_train[1, :].min() - 1, X_train[1, :].max() + 1, n_test)
    X_test = np.asarray(list(itertools.product(x1, x2))).T

    # convert testing features into Shogun representation
    test_features = sg.create_features(X_test)

    # create Gaussian kernel with width = 2.0
    kernel = sg.create_kernel('GaussianKernel', log_width=np.log(2.0))

    # create zero mean function
    mean = sg.ZeroMean()

    # you can easily switch between probit and logit likelihood models
    # by uncommenting/commenting the following lines:

    # create probit likelihood model
    # lik = ProbitLikelihood()

    # create logit likelihood model
    lik = sg.LogitLikelihood()

    # you can easily switch between Laplace and EP approximation by
    # uncommenting/commenting the following lines:

    # specify Laplace approximation inference method
    # inf = SingleLaplacianInferenceMethod(kernel, train_features, mean, train_labels, lik)

    # specify EP approximation inference method
    inf = sg.EPInferenceMethod(kernel, train_features, mean, train_labels, lik)

    # create and train GP classifier, which uses Laplace approximation
    # gp = sg.create_machine('GaussianProcessClassification', inference_method=inf, labels=train_labels)
    gp = sg.GaussianProcessClassification(inf)
    gp.train()

    # get probabilities p(y*=1|x*) for each testing feature x*
    p_test = gp.get_probabilities(test_features)

    # create figure
    plt.title('Training examples, predictive probability and decision boundary')

    # plot training data
    plt.plot(X_train[0, np.argwhere(y_train == 1)], X_train[1, np.argwhere(y_train == 1)], 'ro')
    plt.plot(X_train[0, np.argwhere(y_train == -1)], X_train[1, np.argwhere(y_train == -1)], 'bo')

    # plot decision boundary
    plt.contour(x1, x2, np.reshape(p_test, (n_test, n_test)), levels=[0.5], colors='black')

    # plot probabilities
    plt.pcolor(x1, x2, np.reshape(p_test, (n_test, n_test)))

    # show color bar
    plt.colorbar()

    # show figure
    plt.show()


if __name__ == '__main__':
    X_train, y_train = generate_toy_data()
    gaussian_process_binary_classification_laplace(X_train, y_train)
