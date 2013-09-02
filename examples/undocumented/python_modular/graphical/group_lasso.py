#!/usr/bin/python

import numpy as np
import matplotlib.pyplot as plt
from numpy.random import rand, randn, permutation, multivariate_normal

from modshogun import BinaryLabels, RealFeatures, IndexBlock, IndexBlockGroup, FeatureBlockLogisticRegression


def generate_synthetic_logistic_data(n, p, L, blk_nnz, gcov, nstd):
    # Generates synthetic data for the logistic regression, using the example
    # from [Friedman10]
    # n : # of observations
    # p : # of predictors
    # L : # of blocks
    # blk_nnz : # of non-zero coefs. in each block
    # gcov : correlation within groups
    # nstd : standard deviation of the added noise

    # size of each block (assumed to be an integer)
    pl = p / L

    # generating the coefficients (betas)
    coefs = np.zeros((p, 1))
    for (i, nnz) in enumerate(blk_nnz):
        blkcoefs = np.zeros((pl, 1))
        blkcoefs[0:nnz] = np.sign(rand(nnz, 1) - 0.5)
        coefs[pl * i:pl * (i + 1)] = permutation(blkcoefs)

    # generating the predictors
    mu = np.zeros(p)
    gsigma = gcov * np.ones((pl, pl))
    np.fill_diagonal(gsigma, 1.0)
    Sigma = np.kron(np.eye(L), gsigma)
    # the predictors come from a standard Gaussian multivariate distribution
    X = multivariate_normal(mu, Sigma, n)

    # linear function of the explanatory variables in X, plus noise
    t = np.dot(X, coefs) + randn(n, 1) * nstd
    # applying the logit
    Pr = 1 / (1 + np.exp(-t))
    # The response variable y[i] is a Bernoulli random variable taking 
    # value 1 with probability Pr[i]
    y = rand(n, 1) <= Pr

    # we want each _column_ in X to represent a feature vector
    # y and coefs should be also 1D arrays
    return X.T, y.flatten(), coefs.flatten()


def misclassified_groups(est_coefs, true_coefs, L):
    # Compute the number of groups that are misclassified, i.e. the ones with
    # at least one non-zero coefficient whose estimated coefficients are all 
    # set to zero, or viceversa, as explained in [Friedman10]
    # est_coefs : coefficients estimated by the FBLR
    # true_coefs : the original coefficients of our synthetic example
    # L : number of blocks

    p = est_coefs.shape[0]  # number of predictors
    pl = p / L

    est_nz = est_coefs != 0
    true_nz = true_coefs != 0
    est_blk_nzcount = np.array([sum(est_nz[pl * i:pl * (i + 1)]) for i in xrange(L)])
    true_blk_nzcount = np.array([sum(true_nz[pl * i:pl * (i + 1)]) for i in xrange(L)])
    return np.sum(np.logical_xor(est_blk_nzcount == 0, true_blk_nzcount == 0))


def misclassified_features(est_coefs, true_coefs):
    # Compute the number of individual coefficients that are misclassified,
    # i.e. estimated to be zero when the true coefficient is nonzero or
    # vice-versa, as explained in [Friedman10]
    # est_coefs : coefficients estimated by the FBLR
    # true_coefs : the original coefficients of our synthetic example
    return np.sum(np.logical_xor(est_coefs == 0, true_coefs == 0))


def compute_misclassifications(cls, true_coefs, L, rel_z):
    # Try the given classifier with different values of relative regularization 
    # parameters, store the coefficients and compute the number of groups
    # and features misclassified.
    # INPUTS:
    # - cls : the classifier to try
    # - true_coefs : the original coefficients of our synthetic example
    # - L : number of blocks
    # - rel_z : regularization values to try, they will be in [0,1]
    # OUTPUTS:
    # - est_coefs : array with the estimated coefficients, each row for a 
    #   different value of regularization
    # - misc_groups, misc_feats : see above
    num_z = rel_z.shape[0]
    est_coefs = np.zeros((num_z, true_coefs.shape[0]))
    misc_groups = np.zeros(num_z)
    misc_feats = np.zeros(num_z)
    for (i, z) in enumerate(rel_z):
        cls.set_z(z)
        cls.train()
        est_coefs[i, :] = cls.get_w()
        misc_groups[i] = misclassified_groups(est_coefs[i, :], true_coefs, L)
        misc_feats[i] = misclassified_features(est_coefs[i, :], true_coefs)
    return est_coefs, misc_groups, misc_feats


if __name__ == '__main__':
    print('FeatureBlockLogisticRegression example')

    np.random.seed(956)     # reproducible results

    # default parameters from [Friedman10]
    n = 200
    p = 100
    L = 10
    blk_nnz = [10, 8, 6, 4, 2, 1]
    gcov = 0.2
    nstd = 0.4

    # range of (relative) regularization values to try
    min_z = 0
    max_z = 1
    num_z = 21

    # get the data
    X, y, true_coefs = generate_synthetic_logistic_data(n, p, L, blk_nnz, gcov, nstd)

    # here each column represents a feature vector
    features = RealFeatures(X)
    # we have to convert the labels to +1/-1
    labels = BinaryLabels(np.sign(y.astype(int) - 0.5))

    # SETTING UP THE CLASSIFIERS
    # CLASSIFIER 1: group LASSO
    # build the feature blocks and add them to the block group
    pl = p / L
    block_group = IndexBlockGroup()
    for i in xrange(L):
        block_group.add_block(IndexBlock(pl * i, pl * (i + 1)))

    cls_gl = FeatureBlockLogisticRegression(0.0, features, labels, block_group)
    # with set_regularization(1), the parameter z will indicate the fraction of
    # the maximum regularization to use, and so z is in [0,1]
    # (reference: SLEP manual)
    cls_gl.set_regularization(1)
    cls_gl.set_q(2.0)   # it is the default anyway...

    # CLASSIFIER 2: LASSO (illustrating group lasso with all group sizes = 1)
    block_group_ones = IndexBlockGroup()
    for i in xrange(p):
        block_group_ones.add_block(IndexBlock(i, i + 1))

    cls_l = FeatureBlockLogisticRegression(0.0, features, labels, block_group_ones)
    cls_l.set_regularization(1)
    cls_l.set_q(2.0)

    # trying with different values of (relative) regularization parameters
    rel_z = np.linspace(min_z, max_z, num_z)
    coefs_gl, miscgp_gl, miscft_gl = compute_misclassifications(cls_gl, true_coefs, L, rel_z)
    coefs_l, miscgp_l, miscft_l = compute_misclassifications(cls_l, true_coefs, L, rel_z)

    # Find the best regularization for each classifier
    # for the group lasso: the one that gives the fewest groups misclassified
    best_z_gl = np.argmin(miscgp_gl)
    # for the lasso: the one that gives the fewest features misclassified
    best_z_l = np.argmin(miscft_l)

    # plot the true coefs. and the signs of the estimated coefs.
    fig = plt.figure()
    for (coefs, best_z, name, pos) in zip([coefs_gl, coefs_l], [best_z_gl, best_z_l], ['Group lasso', 'Lasso'], [0, 1]):
        ax = plt.subplot2grid((4, 2), (pos, 0), colspan=2)
        plt.hold(True)
        plt.plot(xrange(p), np.sign(coefs[best_z, :]), 'o', markeredgecolor='none', markerfacecolor='g')
        plt.plot(xrange(p), true_coefs, '^', markersize=7, markeredgecolor='r', markerfacecolor='none', markeredgewidth=1)
        plt.xticks(xrange(0, p + pl, pl))
        plt.yticks([-1, 0, 1])
        plt.xlim((-1, p + 1))
        plt.ylim((-2, 2))
        plt.grid(True)
        # plt.legend(('estimated', 'true'), loc='best')
        plt.title(name)
        plt.xlabel('Predictor [triangles=true coefs], best reg. value = %.2f' % rel_z[best_z])
        plt.ylabel('Coefficient')

    ax = plt.subplot2grid((4, 2), (2, 0), rowspan=2)
    plt.plot(rel_z, miscgp_gl, 'ro-', rel_z, miscgp_l, 'bo-')
    plt.legend(('Group lasso', 'Lasso'), loc='best')
    plt.title('Groups misclassified')
    plt.xlabel('Relative regularization parameter')
    plt.ylabel('# of groups misclassified')

    ax = plt.subplot2grid((4, 2), (2, 1), rowspan=2)
    plt.plot(rel_z, miscft_gl, 'ro-', rel_z, miscft_l, 'bo-')
    plt.legend(('Group lasso', 'Lasso'), loc='best')
    plt.title('Features misclassified')
    plt.xlabel('Relative regularization parameter')
    plt.ylabel('# of features misclassified')

    plt.tight_layout(1.2, 0, 0)
    plt.show()
