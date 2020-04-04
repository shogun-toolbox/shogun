import matplotlib.pyplot as plt
import numpy as np

parameter_list = [[20, 5, 1, 1000, 1, None, 5], [100, 5, 1, 1000, 1, None, 10]]


def classifier_perceptron_graphical(n=100, distance=5, learn_rate=1, max_iter=1000, num_threads=1, seed=None,
                                    nperceptrons=5):
    import shogun as sg

    # 2D data
    _DIM = 2

    np.random.seed(seed)

    # Produce some (probably) linearly separable training data by hand
    # Two Gaussians at a far enough distance
    X = np.array(np.random.randn(_DIM, n)) + distance
    Y = np.array(np.random.randn(_DIM, n))
    label_train_twoclass = np.hstack((np.ones(n), -np.ones(n)))

    fm_train_real = np.hstack((X, Y))
    feats_train = sg.features(fm_train_real)
    labels = sg.labels(label_train_twoclass)

    perceptron = sg.machine('Perceptron', labels=labels, learn_rate=learn_rate, max_iterations=max_iter,
                            initialize_hyperplane=False)

    # Find limits for visualization
    x_min = min(np.min(X[0, :]), np.min(Y[0, :]))
    x_max = max(np.max(X[0, :]), np.max(Y[0, :]))

    y_min = min(np.min(X[1, :]), np.min(Y[1, :]))
    y_max = max(np.max(X[1, :]), np.max(Y[1, :]))

    for i in range(nperceptrons):
        # Initialize randomly weight vector and bias
        perceptron.put('w', np.random.random(2))
        perceptron.put('bias', np.random.random())

        # Run the perceptron algorithm
        perceptron.train(feats_train)

        # Construct the hyperplane for visualization
        # Equation of the decision boundary is w^T x + b = 0
        b = perceptron.get('bias')
        w = perceptron.get('w')

        hx = np.linspace(x_min - 1, x_max + 1)
        hy = -w[1] / w[0] * hx

        plt.plot(hx, -1 / w[1] * (w[0] * hx + b))

    # Plot the two-class data
    plt.scatter(X[0, :], X[1, :], s=40, marker='o', facecolors='none', edgecolors='b')
    plt.scatter(Y[0, :], Y[1, :], s=40, marker='s', facecolors='none', edgecolors='r')

    # Customize the plot
    plt.axis([x_min - 1, x_max + 1, y_min - 1, y_max + 1])
    plt.title('Rosenblatt\'s Perceptron Algorithm')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.show()

    return perceptron


if __name__ == '__main__':
    print('Perceptron graphical')
    classifier_perceptron_graphical(*parameter_list[0])
