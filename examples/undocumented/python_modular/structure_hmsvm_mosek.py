#!/usr/bin/env python

import numpy
import scipy

from scipy import io
data_dict = scipy.io.loadmat('../data/hmsvm_data_large_integer.mat', struct_as_record=False)

parameter_list=[[data_dict]]

def structure_hmsvm_mosek (m_data_dict=data_dict):
	from shogun.Features   import RealMatrixFeatures
	from shogun.Loss       import HingeLoss
	from shogun.Structure  import SequenceLabels, HMSVMModel, Sequence, TwoStateModel, SMT_TWO_STATE
	from shogun.Evaluation import StructuredAccuracy

	try:
		from shogun.Structure import PrimalMosekSOSVM
	except ImportError:
		print("Mosek not available")
		return

	labels_array = m_data_dict['label'][0]

	idxs = numpy.nonzero(labels_array == -1)
	labels_array[idxs] = 0

	labels = SequenceLabels(labels_array, 250, 500, 2)
	features = RealMatrixFeatures(m_data_dict['signal'].astype(float), 250, 500)

	loss = HingeLoss()
	model = HMSVMModel(features, labels, SMT_TWO_STATE, 4)

	sosvm = PrimalMosekSOSVM(model, loss, labels)
	sosvm.train()
	print(sosvm.get_w())

	predicted = sosvm.apply()
	evaluator = StructuredAccuracy()
	acc = evaluator.evaluate(predicted, labels)

	print('Accuracy = %.4f' % acc)

if __name__ == '__main__':
	print("HMSVM Mosek")
	structure_hmsvm_mosek(*parameter_list[0])
