#!/usr/bin/env python

import numpy
import scipy

from scipy import io
data_dict = scipy.io.loadmat('../data/hmsvm_data_large_integer.mat', struct_as_record=False)

parameter_list=[[data_dict]]

def structure_discrete_hmsvm_bmrm (m_data_dict=data_dict):
	from shogun.Features   import RealMatrixFeatures
	from shogun.Structure  import SequenceLabels, HMSVMModel, Sequence, TwoStateModel, SMT_TWO_STATE
	from shogun.Evaluation import StructuredAccuracy
	from shogun.Structure  import DualLibQPBMSOSVM

	labels_array = m_data_dict['label'][0]

	idxs = numpy.nonzero(labels_array == -1)
	labels_array[idxs] = 0

	labels = SequenceLabels(labels_array, 250, 500, 2)
	features = RealMatrixFeatures(m_data_dict['signal'].astype(float), 250, 500)

	num_obs = 4	# given by the data file used
	model = HMSVMModel(features, labels, SMT_TWO_STATE, num_obs)

	sosvm = DualLibQPBMSOSVM(model, labels, 5000.0)
	sosvm.train()
	#print sosvm.get_w()

	predicted = sosvm.apply(features)
	evaluator = StructuredAccuracy()
	acc = evaluator.evaluate(predicted, labels)
	#print('Accuracy = %.4f' % acc)

if __name__ == '__main__':
	print("Discrete HMSVM BMRM")
	structure_discrete_hmsvm_bmrm(*parameter_list[0])
