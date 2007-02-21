from shogun.Features import RealFeatures
from shogun.Kernel import *

acc = 1e-7

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

	#maximal pairwise difference must be smaler than 1e-7
	if max1<1e-7 and max2<1e-7:
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

	#maximal pairwise difference must be smaler than 1e-7
	if max1<1e-7 and max2<1e-7:
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
		test_km=gk.get_kernel_matrix()
	
		max2 =  max(abs(dict['km_test']-gk.get_kernel_matrix()).flat)
	
	except KeyError:
		print 'error in m-file'
		return False	

	#maximal pairwise difference must be smaler than 1e-7
	if max1<1e-7 and max2<1e-7:
		return True

	return False