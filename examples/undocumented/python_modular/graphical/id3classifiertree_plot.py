from modshogun import *
from numpy import array
import pydot

# create data
train_data = array([[1.0, 2.0, 1.0, 3.0, 1.0, 3.0, 2.0, 2.0, 3.0, 1.0, 2.0, 2.0, 3.0, 1.0, 2.0],
[2.0, 1.0, 2.0, 1.0, 2.0, 1.0, 1.0, 2.0, 2.0, 2.0, 1.0, 2.0, 2.0, 2.0, 1.0],
[3.0, 2.0, 3.0, 3.0, 3.0, 2.0, 2.0, 1.0, 3.0, 1.0, 2.0, 1.0, 3.0, 1.0, 2.0],
[1.0, 2.0, 1.0, 2.0, 1.0, 2.0, 1.0, 2.0, 2.0, 1.0, 1.0, 2.0, 2.0, 1.0, 1.0]])

train_labels = array([1.0, 2.0, 1.0, 3.0, 1.0, 2.0, 2.0, 1.0, 3.0, 1.0, 2.0, 1.0, 3.0, 1.0, 2.0])

# wrap features and labels into Shogun objects
feats_train=RealFeatures(train_data)
feats_labels=MulticlassLabels(train_labels)

# ID3 Tree formation
id3=ID3ClassifierTree()
id3.set_labels(feats_labels)
id3.train(feats_train)

# ID3 Tree printing
id3.export_to_graphviz_format()
dot_graph = pydot.graph_from_dot_file('decision_tree.dot')
dot_graph.write_pdf('decision_tree.pdf')
