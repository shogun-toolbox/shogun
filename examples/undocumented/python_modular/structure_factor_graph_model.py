#!/usr/bin/env python

import numpy as np
from modshogun import TableFactorType

# create the factor type with GT parameters
tid = 0
cards = np.array([2,2], np.int32)
w_gt = np.array([0.3,0.5,1.0,0.2,0.05,0.6,-0.2,0.75])
fac_type = TableFactorType(tid, cards, w_gt)

def gen_data(ftype, num_samples, show_data = False):
	from modshogun import Math
	from modshogun import FactorType, Factor, TableFactorType, FactorGraph
	from modshogun import FactorGraphObservation, FactorGraphLabels, FactorGraphFeatures
	from modshogun import MAPInference, TREE_MAX_PROD

	Math.init_random(17)

	samples = FactorGraphFeatures(num_samples)
	labels = FactorGraphLabels(num_samples)

	for i in xrange(num_samples):
		vc = np.array([2,2,2], np.int32)
		fg = FactorGraph(vc)

		data1 = np.array([2.0*Math.random(0.0,1.0)-1.0 for i in xrange(2)])
		vind1 = np.array([0,1], np.int32)
		fac1 = Factor(ftype, vind1, data1)
		fg.add_factor(fac1)

		data2 = np.array([2.0*Math.random(0.0,1.0)-1.0 for i in xrange(2)])
		vind2 = np.array([1,2], np.int32)
		fac2 = Factor(ftype, vind2, data2)
		fg.add_factor(fac2)

		samples.add_sample(fg)
		fg.connect_components()
		fg.compute_energies()

		infer_met = MAPInference(fg, TREE_MAX_PROD)
		infer_met.inference()

		fg_obs = infer_met.get_structured_outputs()
		labels.add_label(fg_obs)

		if show_data:
			state = fg_obs.get_data()
			print state

	return samples, labels

num_samples = 100
samples, labels = gen_data(fac_type, num_samples)

parameter_list = [[samples,labels,w_gt,fac_type]]

def structure_factor_graph_model (tr_samples = samples, tr_labels = labels, w = w_gt, ftype = fac_type):
	from modshogun import FactorGraphModel, MAPInference, TREE_MAX_PROD
	from modshogun import DualLibQPBMSOSVM, LabelsFactory

	# create model
	model = FactorGraphModel(tr_samples, tr_labels, TREE_MAX_PROD, False)
	w_truth = w
	w = np.zeros(8)
	ftype.set_w(w)
	model.add_factor_type(ftype)

	# training
	bmrm = DualLibQPBMSOSVM(model, tr_labels, 1.0)
	bmrm.train()
	#print bmrm.get_w()
	#print w_truth

	# evaluation
	lbs_bmrm = LabelsFactory.to_structured(bmrm.apply())
	acc_loss = 0.0
	ave_loss = 0.0
	for i in xrange(num_samples):
		y_pred = lbs_bmrm.get_label(i)
		y_truth = tr_labels.get_label(i)
		acc_loss = acc_loss + model.delta_loss(y_truth, y_pred)

	ave_loss = acc_loss / num_samples

	print('Average training error is %.4f' % ave_loss)

if __name__ == '__main__':
	print("Factor Graph Model")
	structure_factor_graph_model(*parameter_list[0])
