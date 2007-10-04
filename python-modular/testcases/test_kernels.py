from shogun.Features import RealFeatures, CharFeatures, StringCharFeatures, StringWordFeatures
from shogun.Kernel import *
from shogun.PreProc import *
from shogun.Features import Alphabet,DNA, Labels
from shogun.Classifier import *
from numpy import array, zeros, int32, arange, double, ones
acc = 1e-6

def test_gaussian_kernel(dict):
	try:	
		feat = RealFeatures(dict['traindat'])
		kernel_fun = eval(dict['kernelname'])
		gk=kernel_fun(feat,feat, dict['width_'], dict['size_'])
	
		max1 = max(abs(dict['km_train']-gk.get_kernel_matrix()).flat)


		test_feat = RealFeatures(dict['testdat'])
		gk.init(feat, test_feat)
	
		max2 =  max(abs(dict['km_test']-gk.get_kernel_matrix()).flat)
	except KeyError:
		print 'error in m-file'
		return False
	#maximal pairwise difference must be smaler than acc
	if max1<acc and max2<acc:
		return True

	return False


def test_linear_kernel(dict):
	try:
		feat = RealFeatures(dict['traindat'])
		kernel_fun = eval(dict['kernelname'])
		gk=kernel_fun(feat,feat,eval(dict['bool1']))
		max1 = max(abs(dict['km_train']-gk.get_kernel_matrix()).flat)


		test_feat = RealFeatures(dict['testdat'])
		gk.init(feat, test_feat)
		test_km=gk.get_kernel_matrix()
	
		max2 =  max(abs(dict['km_test']-gk.get_kernel_matrix()).flat)
	except KeyError:
		print 'error in m-file'
		return False
	
	print "max1: ",max1, " max2: ", max2
	#maximal pairwise difference must be smaler than acc
	if max1<acc and max2<acc:
		return True

	return False


def test_chi2_kernel(dict):
	
	try:
		feat = RealFeatures(dict['traindat'])
		kernel_fun = eval(dict['kernelname'])
		gk=kernel_fun(feat,feat,dict['size_'])
	
		max1 = max(abs(dict['km_train'] -gk.get_kernel_matrix()).flat)


		test_feat = RealFeatures(dict['testdat'])
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


def test_sigmoid_kernel(dict):
	try:
		feat = RealFeatures(dict['traindat'])
		kernel_fun = eval(dict['kernelname'])


		gk=kernel_fun(feat,feat, dict['size_'], dict['gamma_'], dict['coef0'])
	
		max1 = max(abs(dict['km_train']-gk.get_kernel_matrix()).flat)


		test_feat = RealFeatures(dict['testdat'])
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

def test_poly_kernel(dict):
	
	try:
		feat = RealFeatures(dict['traindat'])
		kernel_fun = eval(dict['kernelname'])
		gk=kernel_fun(feat,feat,dict['size_'],dict['degree'], eval(dict['inhom']), eval(dict['use_norm']) )
	
		max1 = max(abs(dict['km_train']-gk.get_kernel_matrix()).flat)


		test_feat = RealFeatures(dict['testdat'])
		gk.init(feat, test_feat)
	
		max2 =  max(abs(dict['km_test']-gk.get_kernel_matrix()).flat)
	
	except KeyError:
		print 'error in m-file'
		return False	

	print('%.9f , %.9f'%(max1,max2))
	#maximal pairwise difference must be smaler than acc
	if max1<acc and max2<acc:
		return True

	return False


def test_wdchar_kernel(dict):
	
	try:
		feat = CharFeatures(array(dict['traindat']),eval(dict['alphabet']))
		kernel_fun = eval(dict['kernelname'])
		k=kernel_fun(feat, feat,dict['degree'] )
		max1 = max(abs(dict['km_train']-k.get_kernel_matrix()).flat)

		test_feat = CharFeatures(array(dict['testdat']), eval(dict['alphabet']))
		k.init(feat, test_feat)
	
		max2 =  max(abs(dict['km_test']-k.get_kernel_matrix()).flat)
	
	except KeyError:
		print 'error in m-file'
		return False	

	#maximal pairwise difference must be smaler than acc
	if max1<acc and max2<acc:
		return True

	return False


def test_wd_kernel(dict):
	
	try:
		degree = dict['degree']
		
		stringfeat = StringCharFeatures(eval(dict['alphabet']))
		stringfeat.set_string_features(list(dict['traindat'][0]))

		stringtestfeat = StringCharFeatures(eval(dict['alphabet']))
		stringtestfeat.set_string_features(list(dict['testdat'][0]))
	
		weights = arange(1,degree+1,dtype=double)[::-1]/sum(arange(1,degree+1,dtype=double))
	
		kernel_fun = eval(dict['kernelname'])
		k= kernel_fun(stringfeat, stringfeat, dict['degree'], weights=weights)
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

def test_wds_kernel(dict):
	
	try:
		stringfeat = StringCharFeatures(eval(dict['alphabet']))
		stringfeat.set_string_features(list(dict['traindat'][0]))

		stringtestfeat = StringCharFeatures(eval(dict['alphabet']))
		stringtestfeat.set_string_features(list(dict['testdat'][0]))
		
		kernel_fun = eval(dict['kernelname'])
		k= kernel_fun(stringfeat, stringfeat, dict['degree'],  ones(dict['seqlen'], dtype=int32))
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




def test_cws_kernel(dict):
	
	try:
		stringfeat = StringCharFeatures(eval(dict['alphabet']))
		stringfeat.set_string_features(list(dict['traindat'][0]))

		stringtestfeat = StringCharFeatures(eval(dict['alphabet']))
		stringtestfeat.set_string_features(list(dict['testdat'][0]))
		
 
		wordfeat = StringWordFeatures(stringfeat.get_alphabet());
		wordfeat.obtain_from_char(stringfeat, dict['order']-1, dict['order'], dict['gap'], dict['reverse'])
		
		wordtestfeat = StringWordFeatures(stringtestfeat.get_alphabet());
		wordtestfeat.obtain_from_char(stringtestfeat, dict['order']-1, dict['order'], dict['gap'], dict['reverse'] )

		preproc = SortWordString();
		preproc.init(wordfeat);

		wordfeat.add_preproc(preproc)
		wordfeat.apply_preproc()

		preproc = SortWordString();
		preproc.init(wordtestfeat);

		wordtestfeat.add_preproc(preproc)
		wordtestfeat.apply_preproc()

		kernel_fun = eval(dict['kernelname'])
		k= kernel_fun(wordfeat, wordfeat)
		max1 = max(abs(dict['km_train']-k.get_kernel_matrix()).flat)

		k.init(wordfeat, wordtestfeat)		
		max2 = max(abs(dict['km_test']-k.get_kernel_matrix()).flat)
	
	except KeyError:
		print 'error in m-file'
		return False	

	#maximal pairwise difference must be smaler than acc
	if max1<acc and max2<acc:
		return True

	return False


def test_svm(dict):
	try:	
		feat = RealFeatures(dict['traindat'])
		#kernel_fun = eval(dict['kernelname'])
		gk=GaussianKernel(feat,feat, dict['width_'], dict['size_'])
		numvec = feat.get_num_vectors();
		lab = Labels(double(dict['labels']))
		svm = SVMLight(10,gk,lab)
		svm.train()
		alphas = svm.get_alphas()
		
		max1 = max(alphas-dict['alphas'])


#		test_feat = RealFeatures(dict['testdat'])
#		gk.init(feat, test_feat)
#	
#		max2 =  max(abs(dict['km_test']-gk.get_kernel_matrix()).flat)
	except KeyError:
		print 'error in m-file'
		return False
	#maximal pairwise difference must be smaler than acc
        print('max %i'%max1)
	if max1<acc:
		return True

	return False

