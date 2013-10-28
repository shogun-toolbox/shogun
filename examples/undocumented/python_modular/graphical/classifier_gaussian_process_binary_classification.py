#
# This program is free software you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation either version 3 of the License, or
# (at your option) any later version.
#
# Written (W) 2013 Roman Votyakov
#

from pylab import *
from numpy import *
from itertools import *

def generate_toy_data(n_train=100, mean_a=asarray([0, 0]), std_dev_a=1.0, mean_b=3, std_dev_b=0.5):

    # positive examples are distributed normally
    X1 = (random.randn(n_train, 2)*std_dev_a+mean_a).T

    # negative examples have a "ring"-like form
    r = random.randn(n_train)*std_dev_b+mean_b
    angle = random.randn(n_train)*2*pi
    X2 = array([r*cos(angle)+mean_a[0], r*sin(angle)+mean_a[1]])

    # stack positive and negative examples in a single array
    X_train = hstack((X1,X2))

    # label positive examples with +1, negative with -1
    y_train = zeros(n_train*2)
    y_train[:n_train] = 1
    y_train[n_train:] = -1

    return [X_train, y_train]

def gaussian_process_binary_classification_laplace(X_train, y_train, n_test=50):

    # import all necessary modules from Shogun (some of them require Eigen3)
    try:
        from modshogun import RealFeatures, BinaryLabels, GaussianKernel, \
            LogitLikelihood, ProbitLikelihood, ZeroMean, LaplacianInferenceMethod, \
            EPInferenceMethod, GaussianProcessBinaryClassification
    except ImportError:
        print('Eigen3 needed for Gaussian Processes')
        return

    # convert training data into Shogun representation
    train_features = RealFeatures(X_train)
    train_labels = BinaryLabels(y_train)

    # generate all pairs in 2d range of testing data
    x1 = linspace(X_train[0,:].min()-1, X_train[0,:].max()+1, n_test)
    x2 = linspace(X_train[1,:].min()-1, X_train[1,:].max()+1, n_test)
    X_test = asarray(list(product(x1, x2))).T

    # convert testing features into Shogun representation
    test_features = RealFeatures(X_test)

    # create Gaussian kernel with width = 2.0
    kernel = GaussianKernel(10, 2.0)

    # create zero mean function
    mean = ZeroMean()

    # you can easily switch between probit and logit likelihood models
    # by uncommenting/commenting the following lines:

    # create probit likelihood model
    # lik = ProbitLikelihood()

    # create logit likelihood model
    lik = LogitLikelihood()

    # you can easily switch between Laplace and EP approximation by
    # uncommenting/commenting the following lines:

    # specify Laplace approximation inference method
    # inf = LaplacianInferenceMethod(kernel, train_features, mean, train_labels, lik)

    # specify EP approximation inference method
    inf = EPInferenceMethod(kernel, train_features, mean, train_labels, lik)

    # create and train GP classifier, which uses Laplace approximation
    gp = GaussianProcessBinaryClassification(inf)
    gp.train()

    # get probabilities p(y*=1|x*) for each testing feature x*
    p_test = gp.get_probabilities(test_features)

    # create figure
    figure()
    title('Training examples, predictive probability and decision boundary')

    # plot training data
    plot(X_train[0, argwhere(y_train == 1)], X_train[1, argwhere(y_train == 1)], 'ro')
    plot(X_train[0, argwhere(y_train == -1)], X_train[1, argwhere(y_train == -1)], 'bo')

    # plot decision boundary
    contour(x1, x2, reshape(p_test, (n_test, n_test)), levels=[0.5], colors=('black'))

    # plot probabilities
    pcolor(x1, x2, reshape(p_test, (n_test, n_test)))

    # show color bar
    colorbar()

    # show figure
    show()

if __name__=='__main__':
    [X_train, y_train] = generate_toy_data()
    gaussian_process_binary_classification_laplace(X_train, y_train)
