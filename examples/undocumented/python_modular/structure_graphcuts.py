#!/usr/bin7env python

import numpy as np
import itertools

from modshogun import Factor, TableFactorType, FactorGraph
from modshogun import FactorGraphObservation, FactorGraphLabels, FactorGraphFeatures
from modshogun import FactorGraphModel, GRAPH_CUT
from modshogun import GraphCut
from modshogun import StochasticSOSVM

def generate_data(num_train_samples, len_label, len_feat):
    """ Generate synthetic dataset

        Generate random data following [1]:
        Each example has exactly one label on.
        Each label has 40 related binary features.
        For an example, if label i is on, 4i randomly chosen features are set to 1

        [1] Finley, Thomas, and Thorsten Joachims.
        "Training structural SVMs when exact inference is intractable."
        Proceedings of the 25th international conference on Machine learning. ACM, 2008.

        Args:
            num_train_samples: number of samples
            len_label: label length (10)
            len_feat: feature length (40)

        Returns:
            feats: generated feature matrix
            labels: generated label matrix
    """

    labels = np.zeros((num_train_samples, len_label), np.int32)
    feats = np.zeros((num_train_samples, len_feat), np.int32)

    for k in range(num_train_samples):
        i = k % len_label
        labels[k, i] = 1
        inds_one = np.random.permutation(range(len_feat))
        inds_one = inds_one[:4*(i+1)]
        for j in inds_one:
            feats[k, j] = 1

    return (labels, feats)

def define_factor_types(num_vars, len_feat, edge_table):
    """ Define factor types

        Args:
            num_vars: number of variables in factor graph
            len_feat: length of the feature vector
            edge_table: edge table defines pair-wise node indeces

        Returns:
            v_factor_types: list of all unary and pair-wise factor types
    """
    n_stats = 2 # for binary status
    v_factor_types = {}
    n_edges = edge_table.shape[0]

    # unary factors
    cards_u = np.array([n_stats], np.int32)
    w_u = np.zeros(n_stats*len_feat)
    for i in range(num_vars):
        v_factor_types[i] = TableFactorType(i, cards_u, w_u)

    # pair-wise factors
    cards_pw = np.array([n_stats, n_stats], np.int32)
    w_pw = np.zeros(n_stats*n_stats)
    for j in range(n_edges):
        v_factor_types[j + num_vars] = TableFactorType(j + num_vars, cards_pw, w_pw)

    return v_factor_types

def build_factor_graph_model(labels, feats, factor_types, edge_table, infer_alg = GRAPH_CUT):
    """ Build factor graph model

        Args:
            labels: matrix of labels [num_train_samples*len_label]
            feats: maxtrix of feats [num_train_samples*len_feat]
            factory_types: vectors of all factor types
            edge_table: matrix of pairwised edges, each row is a pair of node indeces
            infer_alg: inference algorithm (GRAPH_CUT)

        Returns:
            labels_fg: matrix of labels in factor graph format
            feats_fg: matrix of features in factor graph format
    """

    labels = labels.astype(np.int32)
    num_train_samples = labels.shape[0]
    num_vars = labels.shape[1]
    num_edges = edge_table.shape[0]
    n_stats = 2

    feats_fg = FactorGraphFeatures(num_train_samples)
    labels_fg = FactorGraphLabels(num_train_samples)

    for i in range(num_train_samples):
        cardinaities = np.array([n_stats]*num_vars, np.int32)
        fg = FactorGraph(cardinaities)

        # add unary factors
        for u in range(num_vars):
            data_u = np.array(feats[i,:], np.float64)
            inds_u = np.array([u], np.int32)
            factor_u = Factor(factor_types[u], inds_u, data_u)
            fg.add_factor(factor_u)

        # add pairwise factors
        for v in range(num_edges):
            data_p = np.array([1.0])
            inds_p = np.array(edge_table[v, :], np.int32)
            factor_p = Factor(factor_types[v + num_vars], inds_p, data_p)
            fg.add_factor(factor_p)

        # add factor graph
        feats_fg.add_sample(fg)

        # add corresponding label
        loss_weights = np.array([1.0/num_vars]*num_vars)
        fg_obs = FactorGraphObservation(labels[i,:], loss_weights)
        labels_fg.add_label(fg_obs)

    return (labels_fg, feats_fg)

def evaluation(labels_pr, labels_gt, model):
    """ Evaluation

        Args:
            labels_pr: predicted label
            labels_gt: ground truth label
            model: factor graph model

        Returns:
            ave_loss: average loss
    """
    num_train_samples = labels_pr.get_num_labels()
    acc_loss = 0.0
    ave_loss = 0.0
    for i in range(num_train_samples):
        y_pred = labels_pr.get_label(i)
        y_truth = labels_gt.get_label(i)
        acc_loss = acc_loss + model.delta_loss(y_truth, y_pred)

    ave_loss = acc_loss / num_train_samples

    return ave_loss

def graphcuts_sosvm(num_train_samples = 20, len_label = 10, len_feat = 40, num_test_samples = 10):
    """ Graph cuts as approximate inference in structured output SVM framework.

        Args:
            num_train_samples: number of training samples
            len_label: number of classes, i.e., size of label space
            len_feat: the dimention of the feature vector
            num_test_samples: number of testing samples
    """
    import time

    # generate synthetic dataset
    (labels_train, feats_train) = generate_data(num_train_samples, len_label, len_feat)

    # compute full-connected edge table
    full = np.vstack([x for x in itertools.combinations(range(len_label), 2)])

    # define factor types
    factor_types = define_factor_types(len_label, len_feat, full)

    # create features and labels for factor graph mode
    (labels_fg, feats_fg) = build_factor_graph_model(labels_train, feats_train, factor_types, full, GRAPH_CUT)

    # create model and register factor types
    model = FactorGraphModel(feats_fg, labels_fg, GRAPH_CUT)

    for i in range(len(factor_types)):
        model.add_factor_type(factor_types[i])

    # Training
    # the 3rd parameter is do_weighted_averaging, by turning this on,
    # a possibly faster convergence rate may be achieved.
    # the 4th parameter controls outputs of verbose training information
    sgd = StochasticSOSVM(model, labels_fg, True, True)
    sgd.set_num_iter(150)
    sgd.set_lambda(0.0001)

    # train
    t0 = time.time()
    sgd.train()
    t1 = time.time()
    w_sgd = sgd.get_w()
    #print "SGD took", t1 - t0, "seconds."

    # training error
    labels_pr = sgd.apply()
    ave_loss = evaluation(labels_pr, labels_fg, model)
    #print('SGD: Average training error is %.4f' % ave_loss)

    # testing error
    # generate synthetic testing dataset
    (labels_test, feats_test) = generate_data(num_test_samples, len_label, len_feat)
    # create features and labels for factor graph mode
    (labels_fg_test, feats_fg_test) = build_factor_graph_model(labels_test, feats_test, factor_types, full, GRAPH_CUT)
    # set features and labels to sgd
    sgd.set_features(feats_fg_test)
    sgd.set_labels(labels_fg_test)
    # test
    labels_pr = sgd.apply()
    ave_loss = evaluation(labels_pr, labels_fg_test, model)
    #print('SGD: Average testing error is %.4f' % ave_loss)

def graphcuts_general():
    """ Graph cuts for general s-t graph optimization.
    """

    num_nodes = 5
    num_edges = 6

    g = GraphCut(num_nodes, num_edges)

    # add termainal-connected edges
    # i.e., SOURCE->node_i and node_i->SINK
    g.add_tweights(0, 4, 0)
    g.add_tweights(1, 2, 0)
    g.add_tweights(2, 8, 0)
    g.add_tweights(2, 0, 4)
    g.add_tweights(3, 0, 7)
    g.add_tweights(4, 0, 5)

    # add node to node edges
    g.add_edge(0, 2, 5, 0)
    g.add_edge(0, 3, 2, 0)
    g.add_edge(1, 2, 6, 0)
    g.add_edge(1, 4, 9, 0)
    g.add_edge(2, 3, 1, 0)
    g.add_edge(2, 4, 3, 0)

    # initialize max-flow algorithm
    g.init_maxflow()

    # compute max flow
    flow = g.compute_maxflow()
    #print("Flow = %f" % flow)

    # print assignment
    #for i in xrange(num_nodes):
    #    print("\nNode %d = %d" % (i, g.get_assignment(i)))

test_general = True
test_sosvm = True
parameter_list = [[test_general, test_sosvm]]

def structure_graphcuts(test_general=True, test_sosvm=True):
    """ Test graph cuts.

        Args:
            test_general: test graph cuts for general s-t graph optimization
            test_sosvm: test graph cuts for structured output svm
    """

    if test_general:
        graphcuts_general()

    if test_sosvm:
        graphcuts_sosvm()

if __name__ == '__main__':
    print("Graph cuts")
    structure_graphcuts(*parameter_list[0])

