#!/usr/bin/env python

"""
This examples shows how to use HierarchicalMultilabelModel for hierarchical multi-label 
classification. The data used:
[1] Image CLEF 2007 competition for annotation of X-Ray images. 
    http://kt.ijs.si/DragiKocev/PhD/resources/doku.php?id=hmc_classification#imageclef07d
"""

from modshogun import MultilabelSOLabels, HierarchicalMultilabelModel
from modshogun import RealFeatures
from modshogun import StochasticSOSVM
from modshogun import StructuredAccuracy, LabelsFactory
import numpy as np
import time


def get_taxonomy(labels):
    """
    Converting the labels to shogun compatible format (i.e. 0, 1, ... num_classes - 1)
    and getting taxonomy of the labels
    """
    labels = labels.split(',')
    num_labels = len(labels)
    # taking the root label into consideration
    num_labels += 1
    shogun_labels = dict()
    taxonomy = np.zeros(num_labels, dtype=np.int32)
    # considering the root_label node index to be 0
    taxonomy[0] = -1
    for i, label in enumerate(labels):
        shogun_labels[label] = i + 1
        try:
            parent_label = label[:-2]
            parent_idx = labels.index(parent_label) + 1
            taxonomy[i + 1] = parent_idx
        except ValueError:
            taxonomy[i + 1] = 0
    return shogun_labels, taxonomy


def get_data_sample(data_sample, shogun_labels):
    """
    Extracting features and labels from a single row of data
    """
    data = data_sample.split(',')
    features = np.array(data[:-1], dtype=np.float64)
    labs = data[-1].split('@')
    # adding the root label
    labels = np.zeros(len(labs) + 1, dtype=np.int32)
    labels[0] = 0
    for i, label in enumerate(labs):
        labels[i + 1] = shogun_labels[label]
    labels.sort()

    return features, labels


def get_data(data, shogun_labels):
    """
    Creating features and labels from the data samples
    """
    num_samples = len(data)
    # considering the root label
    num_classes = len(shogun_labels) + 1
    labels = MultilabelSOLabels(num_samples, num_classes)

    for i, data_sample in enumerate(data):
        feats, labs = get_data_sample(data_sample, shogun_labels)
        try:
            features = np.c_[features, feats]
        except NameError:
            features = feats
        labels.set_sparse_label(i, labs)

    return RealFeatures(features), labels


def get_features_labels(input_file):
    """
    Creating features and labels from the input file (train/test file)
    """
    train_file_lines = map(lambda x: x.strip(), input_file.readlines())

    all_labels = filter(lambda x: 'hierarchical' in x.strip(),
                        train_file_lines)[0].split()[-1]

    shogun_labels, taxonomy = get_taxonomy(all_labels)

    data_index = train_file_lines.index('@DATA')
    features, labels = get_data(train_file_lines[data_index + 1:], shogun_labels)

    return features, labels, taxonomy

if __name__ == '__main__':
    print('Hierarchical Multilabel Classification')

    train_file = open('../../../data/multilabel/image_clef_train.arff')
    test_file = open('../../../data/multilabel/image_clef_test.arff')

    train_features, train_labels, train_taxonomy = get_features_labels(train_file)

    model = HierarchicalMultilabelModel(train_features, train_labels, train_taxonomy)
    sgd = StochasticSOSVM(model, train_labels)
    t1 = time.time()
    sgd.train()
    print('>>> Took %f time for training' % (time.time() - t1))

    test_features, test_labels, test_taxonomy = get_features_labels(test_file)
    assert(test_taxonomy.all() == train_taxonomy.all())

    evaluator = StructuredAccuracy()
    outlabel = LabelsFactory.to_structured(sgd.apply(test_features))

    print('>>> Accuracy of classification = %f' % evaluator.evaluate(outlabel, test_labels))
