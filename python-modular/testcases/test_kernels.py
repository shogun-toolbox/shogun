from shogun.Features import RealFeatures, CharFeatures, StringCharFeatures, StringWordFeatures
from shogun.Kernel import *
from shogun.PreProc import *
from shogun.Features import Alphabet,DNA, Labels
from shogun.Classifier import *
from numpy import array, zeros, int32, arange, double, ones
acc = 1e-6

def test_gaussian(dict):
	try:	
		feat = RealFeatures(dict['data_train'])
		kernel_fun = eval(dict['kernelname'])
		gk=kernel_fun(feat,feat, dict['width_'], dict['size_'])
	
		max1 = max(abs(dict['km_train']-gk.get_kernel_matrix()).flat)

		test_feat = RealFeatures(dict['data_test'])
		gk.init(feat, test_feat)
	
		max2 =  max(abs(dict['km_test']-gk.get_kernel_matrix()).flat)
		print "max1: ",max1, " max2: ", max2
	except KeyError:
		print 'error in m-file'
		return False
	#maximal pairwise difference must be smaler than acc
	if max1<acc and max2<acc:
		return True

	return False


def test_linear(dict):
	try:
		feat = RealFeatures(dict['data_train'])
		kernel_fun = eval(dict['kernelname'])
		gk=kernel_fun(feat,feat,1.0)
		max1 = max(abs(dict['km_train']-gk.get_kernel_matrix()).flat)


		test_feat = RealFeatures(dict['data_test'])
		gk.init(feat, test_feat)
#		test_km=gk.get_kernel_matrix()
	
		max2 =  max(abs(dict['km_test']-gk.get_kernel_matrix()).flat)
		print "max1: ",max1, " max2: ", max2
	except KeyError:
		print 'error in m-file'
		return False
	
	#maximal pairwise difference must be smaler than acc
	if max1<acc and max2<acc:
		return True

	return False


def test_chi2(dict):
	
	try:
		feat = RealFeatures(dict['data_train'])
		kernel_fun = eval(dict['kernelname'])
		gk=kernel_fun(feat,feat,dict['size_'])
	
		max1 = max(abs(dict['km_train'] -gk.get_kernel_matrix()).flat)


		test_feat = RealFeatures(dict['data_test'])
		gk.init(feat, test_feat)
		test_km=gk.get_kernel_matrix()
	
		max2 =  max(abs(dict['km_test']-gk.get_kernel_matrix()).flat)
	
	except KeyError:
		print 'error in m-file'
		return False

	#maximal pairwise difference must be smaler than acc
	if max1<acc and max2<acc:
		return True

	return False


def test_sigmoid(dict):
	try:
		feat = RealFeatures(dict['data_train'])
		kernel_fun = eval(dict['kernelname'])


		gk=kernel_fun(feat,feat, dict['size_'], dict['gamma_'], dict['coef0'])
	
		max1 = max(abs(dict['km_train']-gk.get_kernel_matrix()).flat)


		test_feat = RealFeatures(dict['data_test'])
		gk.init(feat, test_feat)
		test_km=gk.get_kernel_matrix()
	
		max2 =  max(abs(dict['km_test']-gk.get_kernel_matrix()).flat)
	except KeyError:
		print 'error in m-file'
		return False

	#maximal pairwise difference must be smaler than acc
	if max1<acc and max2<acc:
		return True

	return False

def test_poly(dict):
	
	try:
		feat = RealFeatures(dict['data_train'])
		kernel_fun = eval(dict['kernelname'])
		gk=kernel_fun(feat,feat,dict['size_'],dict['degree'], eval(dict['inhom']), eval(dict['use_norm']) )
#		gk=kernel_fun(feat,feat,dict['degree'], eval(dict['inhom']), eval(dict['use_norm'],dict['size_']) )
	
		max1 = max(abs(dict['km_train']-gk.get_kernel_matrix()).flat)


		test_feat = RealFeatures(dict['data_test'])
		gk.init(feat, test_feat)
	
		max2 =  max(abs(dict['km_test']-gk.get_kernel_matrix()).flat)
	
	except KeyError:
		print 'error in m-file'
		return False	

	print('max1 %.12f , max2 %.12f'%(max1,max2))
	#maximal pairwise difference must be smaler than acc
	if max1<acc and max2<acc:
		return True

	return False


def test_weighteddegreestring(dict):
	
	try:
		degree = dict['degree']
		
		stringfeat = StringCharFeatures(eval(dict['alphabet']))
#		import pdb
#		pdb.settrace()
		stringfeat.set_string_features(list(dict['data_train'][0]))

		stringtestfeat = StringCharFeatures(eval(dict['alphabet']))
		stringtestfeat.set_string_features(list(dict['data_test'][0]))
	
		#weights = arange(1,degree+1,dtype=double)[::-1]/sum(arange(1,degree+1,dtype=double))
	
		kernel_fun = eval(dict['kernelname'])
		k= kernel_fun(stringfeat, stringfeat, dict['degree'])
		#k= kernel_fun(stringfeat, stringfeat, dict['degree'], weights=weights)
		max1 = max(abs(dict['km_train']-k.get_kernel_matrix()).flat)

		k.init(stringfeat, stringtestfeat)		
		max2 = max(abs(dict['km_test']-k.get_kernel_matrix()).flat)
	
	except KeyError:
		print 'error in m-file'
		return False	

	#maximal pairwise difference must be smaler than acc
	if max1<acc and max2<acc:
		return True

	return False

def test_weighteddegreepositionstring(dict):
	
	try:
		stringfeat = StringCharFeatures(eval(dict['alphabet']))
		stringfeat.set_string_features(list(dict['data_train'][0]))

		stringtestfeat = StringCharFeatures(eval(dict['alphabet']))
		stringtestfeat.set_string_features(list(dict['data_test'][0]))
		
		kernel_fun = eval(dict['kernelname'])
		k= kernel_fun(stringfeat, stringfeat, dict['degree'],  ones(dict['seqlen'], dtype=int32))
		max1 = max(abs(dict['km_train']-k.get_kernel_matrix()).flat)

		k.init(stringfeat, stringtestfeat)		
		max2 = max(abs(dict['km_test']-k.get_kernel_matrix()).flat)
	
		print "max1: ",max1, " max2: ", max2

	except KeyError:
		print 'error in m-file'
		return False	

	#maximal pairwise difference must be smaler than acc
	if max1<acc and max2<acc:
		return True

	return False




def test_commwordstring(dict):
	
	try:
		stringfeat = StringCharFeatures(eval(dict['alphabet']))
		stringfeat.set_string_features(list(dict['data_train'][0]))

		stringtestfeat = StringCharFeatures(eval(dict['alphabet']))
		stringtestfeat.set_string_features(list(dict['data_test'][0]))
		
 
		wordfeat = StringWordFeatures(stringfeat.get_alphabet());
		wordfeat.obtain_from_char(stringfeat, dict['order']-1, dict['order'], dict['gap'], eval(dict['reverse']))
		
		wordtestfeat = StringWordFeatures(stringtestfeat.get_alphabet());
		wordtestfeat.obtain_from_char(stringtestfeat, dict['order']-1, dict['order'], dict['gap'], eval(dict['reverse']))

		preproc = SortWordString();
		preproc.init(wordfeat);

		wordfeat.add_preproc(preproc)
		wordfeat.apply_preproc()

		#preproc = SortWordString();
		#preproc.init(wordtestfeat);

		wordtestfeat.add_preproc(preproc)
		wordtestfeat.apply_preproc()

		kernel_fun = eval(dict['kernelname'])
		k= kernel_fun(wordfeat, wordfeat)
		max1 = max(abs(dict['km_train']-k.get_kernel_matrix()).flat)

		k.init(wordfeat, wordtestfeat)		
		max2 = max(abs(dict['km_test']-k.get_kernel_matrix()).flat)
	
		print "max1: ",max1, " max2: ", max2
	except KeyError:
		print 'error in m-file'
		return False	

	
	#maximal pairwise difference must be smaler than acc
	if max1<acc and max2<acc:
		return True

	return False


def test_svm_gaussian(dict):
	try:	
		feat = RealFeatures(dict['data_train'])
		#kernel_fun = eval(dict['kernelname'])
		gk=GaussianKernel(feat,feat, dict['width_'], dict['size_'])
#		numvec = feat.get_num_vectors();
		lab = Labels(double(dict['labels']))
		svm = SVMLight(10,gk,lab) #0.1, 1, 10
		svm.train()
		alphas = svm.get_alphas()
		max1 = max(alphas-dict['alphas'])

		#gsv = svm.get_support_vectors()
		#max2 = max(testgsv-dict['alphas']) # eigtl. 0/1 index

		#test_feat = RealFeatures(dict['data_test'])
		#gk.init(feat, test_feat)
		#out = svm.classify().get_labels() #e-4/5 precision

		#bias = svm.get_bias() #e-4/5 precision

		# checken gegen generierte referenz
		#max2 = max(abs(dict['svm_out']-out))
	except KeyError:
		print 'error in m-file'
		return False
	#maximal pairwise difference must be smaler than acc
        print('max %i'%max1)
	if max1<acc:
		return True

	return False

