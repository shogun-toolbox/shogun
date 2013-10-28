#!/usr/bin/env python

def load_compressed_features(fname_features):
	try:
		import gzip
		import numpy
	except ImportError:
		print 'Error importing gzip and/or numpy modules. Please, verify their installation.'
		import sys
		sys.exit(0)

	# load features from a gz compressed file
	file_features = gzip.GzipFile(fname_features)
	str_features = file_features.read()
	file_features.close()

	strlist_features = str_features.split('\n')[:-1] # all but last because the last line also has \n

	# the number of lines in the file is the number of vectors
	num_vectors = len(strlist_features)
	# the number of elements in a line is the number of features
	num_features = len(strlist_features[0].split())
	# memory pre-allocation for the feature matrix
	fm = numpy.zeros((num_vectors, num_features))

	# fill in feature matrix
	for i in xrange(num_vectors):
		try:
			fm[i,:] = map(numpy.float64, strlist_features[i].split())
		except ValuError:
			print 'All the vectors must have the same number of features.'
			import sys
			sys.exit(0)

	return fm

def metric_lmnn_statistics(k=3, fname_features='../../data/fm_train_multiclass_digits.dat.gz', fname_labels='../../data/label_train_multiclass_digits.dat'):
	try:
		from modshogun import LMNN, CSVFile, RealFeatures, MulticlassLabels, MSG_DEBUG
		import matplotlib.pyplot as pyplot
	except ImportError:
		print 'Error importing modshogun or other required modules. Please, verify their installation.'
		return

	features = RealFeatures(load_compressed_features(fname_features).T)
	labels = MulticlassLabels(CSVFile(fname_labels))

#	print 'number of examples = %d' % features.get_num_vectors()
#	print 'number of features = %d' % features.get_num_features()

	assert(features.get_num_vectors() == labels.get_num_labels())

	# train LMNN
	lmnn = LMNN(features, labels, k)
	lmnn.set_correction(100)
#	lmnn.io.set_loglevel(MSG_DEBUG)
	print 'Training LMNN, this will take about two minutes...'
	lmnn.train()
	print 'Training done!'

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
	metric_lmnn_statistics()
