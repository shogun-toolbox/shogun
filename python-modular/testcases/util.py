from shogun.Features import *
from shogun.PreProc import *
from shogun.Distance import *
from numpy import *

def check_accuracy (accuracy, **kwargs):
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

def get_args (input, id):
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

		if i.find('_distance')!=-1: # DistanceKernel
			args[idx]=eval(input[i]+'()')
		else:
			try:
				args[idx]=eval(input[i])
			except TypeError: # no bool
				args[idx]=input[i]

	# weed out superfluous Nones
	return filter(lambda arg: arg is not None, args)

def get_feats_simple (input):
	# have to explicitely set data type for numpy if not real
	data_train=input['data_train'].astype(eval(input['data_type']))
	data_test=input['data_test'].astype(eval(input['data_type']))

	if input['feature_type']=='Byte' or input['feature_type']=='Char':
		alphabet=eval(input['alphabet'])
		train=eval(input['feature_type']+"Features(data_train, alphabet)")
		test=eval(input['feature_type']+"Features(data_test, alphabet)")
	else:
		train=eval(input['feature_type']+"Features(data_train)")
		test=eval(input['feature_type']+"Features(data_test)")

	if (input['name'].find('Sparse')!=-1 or (
		input.has_key('classifier_type') and input['classifier_type']=='linear')):
		sparse_train=eval('Sparse'+input['feature_type']+'Features()')
		sparse_train.obtain_from_simple(train)

		sparse_test=eval('Sparse'+input['feature_type']+'Features()')
		sparse_test.obtain_from_simple(test)

		feats={'train':sparse_train, 'test':sparse_test}
	else:
		feats={'train':train, 'test':test}

	return feats

def get_feats_string (input):
	feats={'train':StringCharFeatures(eval(input['alphabet'])),
		'test':StringCharFeatures(eval(input['alphabet']))}
	feats['train'].set_string_features(list(input['data_train'][0]))
	feats['test'].set_string_features(list(input['data_test'][0]))

	return feats

def get_feats_string_complex (input):
	feats={'train':StringCharFeatures(eval(input['alphabet'])),
		'test':StringCharFeatures(eval(input['alphabet']))}
	feats['train'].set_string_features(list(input['data_train'][0]))
	feats['test'].set_string_features(list(input['data_test'][0]))

	feat=eval('String'+input['feature_type']+"Features(feats['train'].get_alphabet())");
	feat.obtain_from_char(feats['train'], input['order']-1, input['order'],
		input['gap'], eval(input['reverse']))
	if input['feature_type']=='Word':
		preproc=SortWordString();
		preproc.init(feat);
		feat.add_preproc(preproc)
		feat.apply_preproc()
	feats['train']=feat

	feat=eval('String'+input['feature_type']+"Features(feats['train'].get_alphabet())");
	feat.obtain_from_char(feats['test'], input['order']-1, input['order'],
		input['gap'], eval(input['reverse']))
	if input['feature_type']=='Word':
		feat.add_preproc(preproc)
		feat.apply_preproc()
	feats['test']=feat

	return feats


