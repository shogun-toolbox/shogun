from shogun.Features import RealFeatures, CharFeatures, StringCharFeatures, StringWordFeatures
from shogun.Kernel import *
from shogun.PreProc import *
from shogun.Features import *
from shogun.Distance import *
from shogun.Classifier import *
from numpy import *

PREFIX_SVM='svm_'

########################################################################
# helper
########################################################################

def _check_accuracy (accuracy, **kwargs):
	a=double(accuracy)
	output=[]

	for k,v in kwargs.iteritems():
		output.append('%s: %e' % (k, v))
	output.append('accuracy: %e' % accuracy)
	print ', '.join(output)

	for v in kwargs.itervalues():
		if v>a:
			return False

	return True

def _get_args (input, id='kparam'):
	# python dicts are not ordered, so we have to look at the number in
	# the parameter's name and insert items appropriately into an
	# ordered list

	# need to pregenerate list for using indices in loop
	args=len(input)*[None]

	for i in input:
		if i.find(id)==-1:
			continue

		try:
			idx=int(i[len(id)])
		except ValueError:
			raise ValueError, 'Wrong input data %s: "%s"!' % (id, i)

		try:
			args[idx]=eval(input[i])
		except TypeError: # no bool
			args[idx]=input[i]

	# weed out superfluous Nones
	return filter(lambda arg: arg is not None, args)

def _get_feats_simple (input):
	# have to explicitely set data type for numpy if not real
	data_train=input['data_train'].astype(eval(input['data_type']))
	data_test=input['data_test'].astype(eval(input['data_type']))
	feature_type=input['feature_type'].capitalize()

	if feature_type=='Byte' or feature_type=='Char':
		alphabet=eval(input['alphabet'])
		train=eval(feature_type+"Features(data_train, alphabet)")
		test=eval(feature_type+"Features(data_test, alphabet)")
	else:
		train=eval(feature_type+"Features(data_train)")
		test=eval(feature_type+"Features(data_test)")

	if input['name'].find('Sparse')!=-1:
		sparse_train=eval('Sparse'+feature_type+'Features()')
		sparse_train.obtain_from_simple(train)

		sparse_test=eval('Sparse'+feature_type+'Features()')
		sparse_test.obtain_from_simple(test)

		feats={'train':sparse_train, 'test':sparse_test}
	else:
		feats={'train':train, 'test':test}

	return feats

def _get_feats_string (input):
	feats={'train':StringCharFeatures(eval(input['alphabet'])),
		'test':StringCharFeatures(eval(input['alphabet']))}
	feats['train'].set_string_features(list(input['data_train'][0]))
	feats['test'].set_string_features(list(input['data_test'][0]))

	return feats

def _get_feats_wordstring (input):
	feats={'train':StringCharFeatures(eval(input['alphabet'])),
		'test':StringCharFeatures(eval(input['alphabet']))}
	feats['train'].set_string_features(list(input['data_train'][0]))
	feats['test'].set_string_features(list(input['data_test'][0]))

	wordfeat=StringWordFeatures(feats['train'].get_alphabet());
	wordfeat.obtain_from_char(feats['train'], input['order']-1, input['order'],
		input['gap'], eval(input['reverse']))
	preproc = SortWordString();
	preproc.init(wordfeat);
	wordfeat.add_preproc(preproc)
	wordfeat.apply_preproc()
	feats['train']=wordfeat

	wordfeat=StringWordFeatures(feats['test'].get_alphabet());
	wordfeat.obtain_from_char(feats['test'], input['order']-1, input['order'],
		input['gap'], eval(input['reverse']))
	#preproc = SortWordString();
	#preproc.init(wordfeat);
	wordfeat.add_preproc(preproc)
	wordfeat.apply_preproc()
	feats['test']=wordfeat

	return feats

########################################################################
# kernel computation
########################################################################

def _kernel (input, feats):
	kfun=eval(input['name']+'Kernel')
	args=_get_args(input)

	k=kfun(feats['train'], feats['train'], *args)
	train=max(abs(input['km_train']-k.get_kernel_matrix()).flat)
	k.init(feats['train'], feats['test'])
	test=max(abs(input['km_test']-k.get_kernel_matrix()).flat)

	return _check_accuracy(input['accuracy'], train=train, test=test)

def _add_subkernels (subkernels):
	for idx, sk in enumerate(subkernels):
		fun=eval(sk['name']+'Kernel')
		args=_get_args(sk)
		subkernels[idx]['kernel']=fun(*args)
	return subkernels

def _get_subkernels (input):
	subkernels=len(input)*[None]
	len_subkernel=len('subkernel')

	for i in input:
		if i.find('subkernel')==-1:
			continue

		try:
			idx=int(i[len_subkernel])
		except ValueError:
			raise ValueError, 'Wrong input data subkernel: "%s"!' % i

		# get item's name
		item=i[i.find('_')+1:]

		# weird behaviour of python if subkernels is inited with {}, so
		# have to do this:
		if subkernels[idx] is None:
			subkernels[idx]={}

		subkernels[idx][item]=input[i]

	# weed out empty subkernels
	subkernels=filter(lambda x: x is not None, subkernels)
	return _add_subkernels(subkernels)

def _kernel_combined (input):
	kernel=CombinedKernel()
	feats={'train':CombinedFeatures(), 'test':CombinedFeatures()}

	subkernels=_get_subkernels(input)
	for sk in subkernels:
		sk['alphabet']=input['alphabet']
		feats_sk=eval('_get_feats_'+sk['feature_class']+'(sk)')
		kernel.append_kernel(sk['kernel'])
		feats['train'].append_feature_obj(feats_sk['train'])
		feats['test'].append_feature_obj(feats_sk['test'])

	return _kernel_subkernels(input, feats, kernel)

def _kernel_auc (input):
	sk=_get_subkernels(input)[0]
	feats_sk=eval('_get_feats_'+sk['feature_class']+'(sk)')
	sk['kernel'].init(feats_sk['train'], feats_sk['test'])

	feats={
		'train':WordFeatures(input['data_train'].astype(eval(input['data_type']))),
		'test':WordFeatures(input['data_test'].astype(eval(input['data_type'])))}
	kernel=AUCKernel(10, sk['kernel'])

	return _kernel_subkernels(input, feats, kernel)

def _kernel_subkernels (input, feats, kernel):
	kernel.init(feats['train'], feats['train'])
	train=max(abs(input['km_train']-kernel.get_kernel_matrix()).flat)
	kernel.init(feats['train'], feats['test'])
	test=max(abs(input['km_test']-kernel.get_kernel_matrix()).flat)

	return _check_accuracy(input['accuracy'],
		train=train, test=test)

def _kernel_svm (input):
	feats={'train':RealFeatures(input['data_train']),
		'test':RealFeatures(input['data_test'])}
	args=_get_args(input)

	kfun=eval(input['name']+'Kernel')
	k=kfun(feats['train'], feats['train'], *args)
	k.parallel.set_num_threads(input['num_threads'])
	l=Labels(double(input['labels']))
	svm=SVMLight(input['C'], k, l)
	svm.parallel.set_num_threads(input['num_threads'])
	svm.set_epsilon(input['epsilon'])
	svm.set_tube_epsilon(input['tube_epsilon'])
	svm.train()

	if input['num_threads']==1:
		check_alphas=max(abs(svm.get_alphas()-input['alphas']))
		check_bias=abs(svm.get_bias()-input['bias'])
		check_sv=max(abs(svm.get_support_vectors()-input['support_vectors']))
	else: # lower accuracy, less checks if parallel
		accuracy=1e-4
		check_alphas=0.
		check_bias=0.
		check_sv=0.

	k.init(feats['train'], feats['test'])
	check_classified=max(abs(svm.classify().get_labels()-input['classified']))

	return _check_accuracy(input['accuracy'],
		alphas=check_alphas, bias=check_bias, sv=check_sv,
		classified=check_classified)

########################################################################
# public
########################################################################

def test (input):
	if input['name']=='Combined':
		return _kernel_combined(input)
	elif input['name']=='AUC':
		return _kernel_auc(input)
	elif input['name'].startswith(PREFIX_SVM):
		input['name']=input['name'][len(PREFIX_SVM):]
		return _kernel_svm(input)
	else:
		fun=eval('_get_feats_'+input['feature_class'])
		feats=fun(input)
		return _kernel(input, feats)

