def io ():
	print 'Features IO'
	import numpy
	from shogun.Features import RealFeatures, Labels
	from shogun.Kernel import GaussianKernel
	from shogun.Library import AsciiFile, BinaryFile

	feats=RealFeatures(fm_train_real)
	feats2=RealFeatures()

	f=BinaryFile("fm_train_real.bin","w")
	feats.save(f)

	f=AsciiFile("fm_train_real.ascii","w")
	feats.save(f)

	f=BinaryFile("fm_train_real.bin")
	feats2.load(f)
	print "diff binary", numpy.max(numpy.abs(feats2.get_feature_matrix().flatten()-fm_train_real.flatten()))

	f=AsciiFile("fm_train_real.ascii")
	feats2.load(f)
	print "diff ascii", numpy.max(numpy.abs(feats2.get_feature_matrix().flatten()-fm_train_real.flatten()))

	lab=Labels(numpy.array([1.0,2.0,3.0]))
	lab2=Labels()
	f=AsciiFile("label_train_twoclass.ascii","w")
	lab.save(f)

	f=BinaryFile("label_train_twoclass.bin","w")
	lab.save(f)

	f=AsciiFile("label_train_twoclass.ascii")
	lab2.load(f)

	f=BinaryFile("label_train_twoclass.bin")
	lab2.load(f)

if __name__=='__main__':
	from tools.load import LoadMatrix
	lm=LoadMatrix()
	fm_train_real=lm.load_numbers('../data/fm_train_real.dat')
	label_train_twoclass=lm.load_numbers('../data/label_train_twoclass.dat')
	io()
