#!/usr/bin/env python

import numpy as np
import scipy

from scipy import io
data_dict = scipy.io.loadmat('../data/hmsvm_data_large_integer.mat', struct_as_record=False)

parameter_list=[[data_dict]]

def structure_discrete_hmsvm_mosek (m_data_dict=data_dict):
	import shogun as sg

	try:
		_ = sg.machine("PrimalMosekSOSVM")
	except:
		print("Mosek not available")
		return

	labels_array = m_data_dict['label'][0]

	idxs = np.nonzero(labels_array == -1)
	labels_array[idxs] = 0

	labels = sg.SequenceLabels(labels_array, 250, 500, 2)
	features = sg.RealMatrixFeatures(m_data_dict['signal'].astype(float), 250, 500)

	num_obs = 4	# given by the data file used
	model = sg.structured_model("HMSVMModel", features=features, labels=labels, 
								state_model_type=SMT_TWO_STATE, num_obs=num_obs)

	sosvm = sg.machine("PrimalMosekSOSVM", model=model, labels=labels)
	sosvm.train()

	predicted = sosvm.apply()
	evaluator = sg.evaluation("StructuredAccuracy")
	acc = evaluator.evaluate(predicted, labels)

if __name__ == '__main__':
	print("Discrete HMSVM Mosek")
	structure_discrete_hmsvm_mosek(*parameter_list[0])
