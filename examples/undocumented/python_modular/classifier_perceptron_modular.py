def perceptron ():
	print 'Perceptron'

	from shogun.Features import RealFeatures, Labels
	from shogun.Classifier import Perceptron

	feats_train=RealFeatures(fm_train_real)
	feats_test=RealFeatures(fm_test_real)

	learn_rate=1.
	max_iter=1000
	num_threads=1
	labels=Labels(label_train_twoclass)

	perceptron=Perceptron(feats_train, labels)
	perceptron.set_learn_rate(learn_rate)
	perceptron.set_max_iter(max_iter)
	# only guaranteed to converge for separable data
	perceptron.train()

	perceptron.set_features(feats_test)
	perceptron.classify().get_labels()

if __name__=='__main__':
	from tools.load import LoadMatrix
	lm=LoadMatrix()
	fm_train_real=lm.load_numbers('../data/fm_train_real.dat')
	fm_test_real=lm.load_numbers('../data/fm_test_real.dat')
	label_train_twoclass=lm.load_labels('../data/label_train_twoclass.dat')
	perceptron()
