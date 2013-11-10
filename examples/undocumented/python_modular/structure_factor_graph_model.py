#!/usr/bin/env python

import numpy as np
from modshogun import TableFactorType

# create the factor type with GT parameters
tid = 0
cards = np.array([2,2], np.int32)
w_gt = np.array([0.3,0.5,1.0,0.2,0.05,0.6,-0.2,0.75])
fac_type = TableFactorType(tid, cards, w_gt)

tid_u = 1
cards_u = np.array([2], np.int32)
w_gt_u = np.array([0.5,0.8,1.0,-0.3])
fac_type_u = TableFactorType(tid_u, cards_u, w_gt_u)

tid_b = 2
cards_b = np.array([2], np.int32)
w_gt_b = np.array([0.8, -0.8])
fac_type_b = TableFactorType(tid_b, cards_b, w_gt_b)

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
		fac1 = Factor(ftype[0], vind1, data1)
		fg.add_factor(fac1)

		data2 = np.array([2.0*Math.random(0.0,1.0)-1.0 for i in xrange(2)])
		vind2 = np.array([1,2], np.int32)
		fac2 = Factor(ftype[0], vind2, data2)
		fg.add_factor(fac2)

		data3 = np.array([2.0*Math.random(0.0,1.0)-1.0 for i in xrange(2)])
		vind3 = np.array([0], np.int32)
		fac3 = Factor(ftype[1], vind3, data3)
		fg.add_factor(fac3)

		data4 = np.array([2.0*Math.random(0.0,1.0)-1.0 for i in xrange(2)])
		vind4 = np.array([1], np.int32)
		fac4 = Factor(ftype[1], vind4, data4)
		fg.add_factor(fac4)

		data5 = np.array([2.0*Math.random(0.0,1.0)-1.0 for i in xrange(2)])
		vind5 = np.array([2], np.int32)
		fac5 = Factor(ftype[1], vind5, data5)
		fg.add_factor(fac5)

		data6 = np.array([1.0])
		vind6 = np.array([0], np.int32)
		fac6 = Factor(ftype[2], vind6, data6)
		fg.add_factor(fac6)

		data7 = np.array([1.0])
		vind7 = np.array([2], np.int32)
		fac7 = Factor(ftype[2], vind7, data7)
		fg.add_factor(fac7)

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


w_all = [w_gt,w_gt_u,w_gt_b]
ftype_all = [fac_type,fac_type_u,fac_type_b]

num_samples = 10
samples, labels = gen_data(ftype_all, num_samples)

parameter_list = [[samples,labels,w_all,ftype_all]]

def structure_factor_graph_model(tr_samples = samples, tr_labels = labels, w = w_all, ftype = ftype_all):
	from modshogun import SOSVMHelper, LabelsFactory
	from modshogun import FactorGraphModel, MAPInference, TREE_MAX_PROD
	from modshogun import DualLibQPBMSOSVM, StochasticSOSVM

	# create model
	model = FactorGraphModel(tr_samples, tr_labels, TREE_MAX_PROD, False)
	w_truth = [w[0].copy(), w[1].copy(), w[2].copy()]
	w[0] = np.zeros(8)
	w[1] = np.zeros(4)
	w[2] = np.zeros(2)
	ftype[0].set_w(w[0])
	ftype[1].set_w(w[1])
	ftype[2].set_w(w[2])
	model.add_factor_type(ftype[0])
	model.add_factor_type(ftype[1])
	model.add_factor_type(ftype[2])

	# --- training with BMRM ---
	bmrm = DualLibQPBMSOSVM(model, tr_labels, 0.01)
	#bmrm.set_verbose(True)
	bmrm.train()
	#print 'learned weights:'
	#print bmrm.get_w()
	#print 'ground truth weights:'
	#print w_truth

	# evaluation
	eva_bmrm = bmrm.apply()
	lbs_bmrm = LabelsFactory.to_structured(eva_bmrm)
	acc_loss = 0.0
	ave_loss = 0.0
	for i in xrange(num_samples):
		y_pred = lbs_bmrm.get_label(i)
		y_truth = tr_labels.get_label(i)
		acc_loss = acc_loss + model.delta_loss(y_truth, y_pred)

	ave_loss = acc_loss / num_samples

	#print('BMRM: Average training error is %.4f' % ave_loss)

	# show primal objs and dual objs
	#hbm = bmrm.get_helper()
	#print hbm.get_primal_values()
	#print hbm.get_eff_passes()
	#print hbm.get_train_errors()

	# --- training with SGD ---
	sgd = StochasticSOSVM(model, tr_labels)
	#sgd.set_verbose(True)
	sgd.set_lambda(0.01)
	sgd.train()

	# evaluation
	#print('SGD: Average training error is %.4f' % SOSVMHelper.average_loss(sgd.get_w(), model))
	#hp = sgd.get_helper()
	#print hp.get_primal_values()
	#print hp.get_eff_passes()
	#print hp.get_train_errors()

if __name__ == '__main__':
	print("Factor Graph Model")
	structure_factor_graph_model(*parameter_list[0])
