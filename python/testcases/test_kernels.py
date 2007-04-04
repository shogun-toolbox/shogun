
from numpy import array,transpose, reshape
import sg
eps= 1e-6

def test_gaussian_kernelalt(dict):
	try:	
		sg.set_features("TRAIN", dict['traindat'])
		bool = 0
		#if dict['bool1'].startswith('True'):
			#bool = 1
		sg.send_command('set_kernel GAUSSIAN REAL %i %f'%(dict['size_'],dict['width_']))
		#print "size: ", dict['size_'],"width: ", dict['width_']
		sg.send_command('init_kernel TRAIN')
		#print "train:    ", dict['traindat']
		f = sg.get_features("TRAIN")
		#print "features: ", reshape(f,(f.shape[1],-1))
		k = sg.get_kernel_matrix()
		#print "k: ",k
		#print "km_train: ",dict['km_train']
		max1 = max(abs(dict['km_train']-k).flat)
		print "max1: ", max1
		sg.set_features("TEST",dict['testdat'])
		sg.send_command('init_kernel TEST')
		k2 = sg.get_kernel_matrix()
		#k2 =  transpose(reshape(k2,(k2.shape[1],-1)))
		#print "km_test: ", dict['km_test']
		#print "k2: ", k2
		max2 =  max(abs(dict['km_test']-k2).flat)
		
		print "max2: ", max2
	except KeyError:
		print 'error in m-file'
		return False
	#maximal pairwise difference must be smaler than acc

	if max1<eps and max2<eps:
		return True

	return False

def test_gaussian_kernel(dict):
	try:	
		sg.set_features("TRAIN", dict['traindat'])
		sg.send_command('set_kernel GAUSSIAN REAL %i %f'%(dict['size_'],dict['width_']))
		sg.send_command('init_kernel TRAIN')
		k = sg.get_kernel_matrix()
		max1 = max(abs(dict['km_train']-k).flat)

		sg.set_features("TEST",dict['testdat'])
		sg.send_command('init_kernel TEST')
		k2 = sg.get_kernel_matrix()
		max2 =  max(abs(dict['km_test']-k2).flat)
	except KeyError:
		print 'error in m-file'
		return False

	#maximal pairwise difference must be smaler than eps
	if max1<eps and max2<eps:
		return True
	return False

def test_linear_kernel(dict):
	try:	
		sg.set_features("TRAIN", dict['traindat'])
		bool = 0
		if dict['bool1'].startswith('True'):
			bool = 1
		sg.send_command('set_kernel LINEAR REAL %i'%(bool))
		sg.send_command('init_kernel TRAIN')
		k = sg.get_kernel_matrix()
		max1 = max(abs(dict['km_train']-k).flat)

		sg.set_features("TEST",dict['testdat'])
		sg.send_command('init_kernel TEST')
		k2 = sg.get_kernel_matrix()
		max2 =  max(abs(dict['km_test']-k2).flat)
	except KeyError:
		print 'error in m-file'
		return False

	#maximal pairwise difference must be smaler than eps
	if max1<eps and max2<eps:
		return True
	return False

def test_chi2_kernel(dict):
	try:	
		sg.set_features("TRAIN", dict['traindat'])
		sg.send_command('set_kernel CHI2 REAL %i '%(dict['size_']))
		sg.send_command('init_kernel TRAIN')
		k = sg.get_kernel_matrix()
		max1 = max(abs(dict['km_train']-k).flat)

		sg.set_features("TEST",dict['testdat'])
		sg.send_command('init_kernel TEST')
		k2 = sg.get_kernel_matrix()
		max2 =  max(abs(dict['km_test']-k2).flat)
	except KeyError:
		print 'error in m-file'
		return False

	#maximal pairwise difference must be smaler than eps
	if max1<eps and max2<eps:
		return True
	return False
def test_sigmoid_kernel(dict):
	try:	
		sg.set_features("TRAIN", dict['traindat'])
		sg.send_command('set_kernel SIGMOID REAL %i %f %f'%(dict['size_'],dict['gamma_'],dict['coef0']))
		sg.send_command('init_kernel TRAIN')
		k = sg.get_kernel_matrix()
		max1 = max(abs(dict['km_train']-k).flat)

		sg.set_features("TEST",dict['testdat'])
		sg.send_command('init_kernel TEST')
		k2 = sg.get_kernel_matrix()
		max2 =  max(abs(dict['km_test']-k2).flat)
	except KeyError:
		print 'error in m-file'
		return False

	#maximal pairwise difference must be smaler than eps
	if max1<eps and max2<eps:
		return True
	return False
def test_poly_kernel(dict):
	try:	
		sg.set_features("TRAIN", dict['traindat'])
		inhom = 0
		if dict['inhom'].startswith('True'):
			inhom = 1
		use_norm = 0
		if dict['use_norm'].startswith('True'):
			use_norm = 1
		sg.send_command('set_kernel POLY REAL %i %i %i %i'%(dict['size_'],dict['degree'],inhom, use_norm))
		sg.send_command('init_kernel TRAIN')
		k = sg.get_kernel_matrix()
		max1 = max(abs(dict['km_train']-k).flat)

		sg.set_features("TEST",dict['testdat'])
		sg.send_command('init_kernel TEST')
		k2 = sg.get_kernel_matrix()
		max2 =  max(abs(dict['km_test']-k2).flat)
	except KeyError:
		print 'error in m-file'
		return False

	#maximal pairwise difference must be smaler than eps
	if max1<eps and max2<eps:
		return True
	return False

def test_wdchar_kernel(dict):
	try:	
		sg.set_features("TRAIN", dict['traindat'], "DNA")
		#sg.send_command('set_kernel WEIGHTEDDEGREE CHAR %i'%(dict['degree']))
		#sg.send_command('init_kernel TRAIN')
		#k = sg.get_kernel_matrix()
		#max1 = max(abs(dict['km_train']-k).flat)

		#sg.set_features("TEST",dict['testdat'], 'DNA')
		#sg.send_command('init_kernel TEST')
		#k2 = sg.get_kernel_matrix()
		#max2 =  max(abs(dict['km_test']-k2).flat)
	except KeyError:
		print 'error in m-file'
		return False

	#maximal pairwise difference must be smaler than eps
	#if max1<eps and max2<eps:
		#return True
	#return False
def test_wdchar_kernel_alt(dict):
	
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
	if max1<eps and max2<eps:
		return True

	return False