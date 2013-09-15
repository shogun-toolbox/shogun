#!/usr/bin/env python

def load_data(data_fname='../../data/multiclass_digits.mat', num_examples=2000):
	try:
		import scipy
		from scipy import io
		import numpy
		from modshogun import RealFeatures, MulticlassLabels
	except ImportError:
		return

	data_dict = scipy.io.loadmat(data_fname, struct_as_record=False)
	features = RealFeatures(data_dict['xTr'][:,:num_examples].astype(numpy.float64))
	labels = MulticlassLabels(data_dict['yTr'][0][:num_examples].astype(numpy.float64))

	assert(features.get_num_vectors() == labels.get_num_labels())

# 	print 'number of examples = %d' % features.get_num_vectors()
# 	print 'number of features = %d' % features.get_num_features()

	return features, labels

def metric_lmnn_statistics(features, labels, k=3):
	try:
		from modshogun import LMNN, MSG_DEBUG
		import matplotlib.pyplot as pyplot
	except ImportError:
		return

	# train LMNN
	lmnn = LMNN(features, labels, k)
	lmnn.set_correction(50)
# 	lmnn.io.set_loglevel(MSG_DEBUG)
	lmnn.train()

	# plot objective obtained during training
	statistics = lmnn.get_statistics()
	
	pyplot.plot(statistics.obj.get())
	pyplot.grid(True)
	pyplot.xlabel('Iterations')
	pyplot.ylabel('LMNN objective')
	pyplot.title('LMNN objective during training for the multiclass digits data set')

	pyplot.show()

if __name__=='__main__':
	print('LMNN objective')
	features, labels = load_data()
	metric_lmnn_statistics(features, labels)
