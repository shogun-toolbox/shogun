"""Common operations related to file handling"""

import os
from numpy import int, long, float, double, ushort, uint16, ubyte, short, matrix, int32, int64, uint32, uint64

import featop
import dataop
import category

DIR_OUTPUT='data'
EXT_OUTPUT='.m'


def _get_matrix (name, kmatrix):
	"""Converts a numpy matrix into a matrix digestable by e.g. matlab.

	@param name Name of the matrix
	@param kmatrix The matrix
	@return String which contains a matrix digestable by e.g. matlab
	"""

	line=list()
	list_numbers=(int, long, float, double, ubyte, ushort, short, uint16, int32, int64, uint32, uint64)
	matrix=[]
	is_string=True

	try:
		# assume, all elements have same data type
		if isinstance(kmatrix[0, 0], list_numbers):
			is_string=False

		for x in range(kmatrix.shape[0]):
			for y in range(kmatrix.shape[1]):
				if is_string:
					line.append("'%s'" % kmatrix[x, y])
				else:
					line.append('%.9g' % kmatrix[x, y])
			matrix.append(', '.join(line))
			line=list()
	except IndexError:
		if isinstance(kmatrix[0], list_numbers):
			is_string=False

		for x in range(kmatrix.shape[0]):
			if is_string:
				line.append("'%s'" % kmatrix[x])
			else:
				line.append('%.9g' % kmatrix[x])
		matrix.append(', '.join(line))
		line=list()

	matrix=';'.join(matrix)

	if is_string:
		matrix=''.join([name, ' = {', matrix, '}'])
	else:
		matrix=''.join([name, ' = [', matrix, ']'])

	return matrix.replace('\n', '')


def _is_excluded_from_filename (key):
	"""
	Determine if given key's value shall not be part of the filename.

	@param key name of the value to check for.
	@return true if to be excluded, false otherwise
	"""

	if (key.find('feature_')!=-1 or
		key.find('accuracy')!=-1 or
		key.find('data_')!=-1 or
		key.find('normalizer')!=-1 or
		key=='name' or
		key=='init_random' or
		key=='regression_type' or
		key=='regression_bias' or
		key=='regression_alpha_sum' or
		key=='regression_sv_sum' or
		key=='distribution_likelihood' or
		key=='distribution_derivatives' or
		key=='distribution_best_path' or
		key=='distribution_best_path_state' or
		key=='classifier_bias' or
		key=='classifier_label_type' or
		key=='classifier_alpha_sum' or
		key=='classifier_sv_sum' or
		key=='classifier_type'):
		return True
	else:
		return False


def _get_filename (catID, out):
	"""
	Return filename for testcase data's output.

	@param catID ID of the category
	@param out data to be written into file
	@return string with the filename
	"""

	params=[]

	name=category.get_name(catID)
	for key, val in out.iteritems():
		if _is_excluded_from_filename(key):
			continue
		if key==name:
			continue

		cname=val.__class__.__name__
		if cname=='bool' or cname=='float' or cname=='int' or cname=='str':
			val=str(val)
			val=val.replace('-', 'n')
			val=val.replace('+', 'p')
			params.append(val)

	params='_'.join(params).replace('.', '')
	if len(params)>0:
		params='_'+params
	else: # otherwise problems with one interface (FIXME: find out which)
		params='_fnord'

	dir=DIR_OUTPUT+os.sep+category.get_as_string(catID)+os.sep
	return dir+out[name]+params+EXT_OUTPUT


def _loop_args(args, prefix):
	"""
	Loop through all arguments in given dict and add to appropriately to
	output data.

	@param params various clustering parameters
	@param prefix prefix for parameter's name, e.g. 'preproc_'
	@return dict containing testcase data ready to be written to file
	"""

	out={}
	for i in xrange(len(args['key'])):
		try:
			name=prefix+'arg'+str(i)+'_'+args['key'][i]
		except IndexError:
			break

		cname=args['val'][i].__class__.__name__
		if cname=='int' or cname=='float' or cname=='bool' or cname=='str':
			out[name]=args['val'][i]
		else:
			out[name]=cname

	return out


def _get_output_classifier (params, prefix=''):
	"""
	Classifier-specific gathering of output data

	@param params various classifier parameters
	@param prefix prefix for parameter's name, e.g. 'classifier_'
	@return dict containing testcase data ready to be written to file
	"""

	out={}
	for key, val in params.iteritems():
		if key!='data' and val is not None:
			out[prefix+key]=val

	return out


def _get_output_clustering (params, prefix=''):
	"""
	Clustering-specific gathering of output data

	@param params various clustering parameters
	@param prefix prefix for parameter's name, e.g. 'clustering_'
	@return dict containing testcase data ready to be written to file
	"""

	out={}
	for key, val in params.iteritems():
		out[prefix+key]=val

	return out


def _get_output_distance (params, prefix=''):
	"""
	Distance-specific gathering of outdata

	@param params various distance parameters
	@param prefix prefix for parameter's name, e.g. 'subkernel'
	@return dict containing testcase data ready to be written to file
	"""

	if not params.has_key('args'):
		return {}

	return _loop_args(params['args'], prefix)


def _get_output_distribution (params, prefix=''):
	"""
	Clustering-specific gathering of output data

	@param params various clustering parameters
	@param prefix prefix for parameter's name, e.g. 'clustering_'
	@return dict containing testcase data ready to be written to file
	"""

	out={}
	for key, val in params.iteritems():
		if key!='data' and val is not None:
			out[prefix+key]=val

	return out


def _get_output_kernel (params, prefix=''):
	"""
	Kernel-specific gathering of outdata

	@param params various kernel parameters
	@param prefix prefix for parameter's name, e.g. 'subkernel'
	@return dict containing testcase data ready to be written to file
	"""

	out={}

	if params.has_key('normalizer'):
		out[prefix+'normalizer']=params['normalizer'].__class__.__name__

	if not params.has_key('args'):
		return out

	if params['name']=='AUC':
		# remove element 'subkernel'
		params['args']['key']=(params['args']['key'][0],)

	out.update(_loop_args(params['args'], prefix))
	return  out


def _get_output_preproc (params, prefix=''):
	"""
	Preproc-specific gathering of output data

	@param params various preproc parameters
	@param prefix prefix for parameter's name, e.g. 'preproc_'
	@return dict containing testcase data ready to be written to file
	"""

	if not params.has_key('args'):
		return {}

	return _loop_args(params['args'], prefix)


def _get_output_regression (params, prefix=''):
	"""
	Regression-specific gathering of output data

	@param params various regression parameters
	@param prefix prefix for parameter's name, e.g. 'regression_'
	@return dict containing testcase data ready to be written to file
	"""

	out={}
	for key, val in params.iteritems():
		out[prefix+key]=val

	return out



############################################################################
# public
############################################################################

def write (catID, out):
	"""
	Write given testcase data to a file.

	@param cat ID of the category, like category.KERNEL
	@param out data to be written into file
	@return Success of operation
	"""

	fnam=_get_filename(catID, out)
	print 'Writing for '+category.get_as_string(catID).upper()+': '+ \
		os.path.basename(fnam)

	dirname=os.path.dirname(fnam)
	if not os.path.isdir(dirname):
		os.mkdir(dirname)

	mfile=open(fnam, mode='w')
	for key, val in out.iteritems():
		cname=val.__class__.__name__
		if cname=='bool' or cname=='str':
			mfile.write("%s = '%s';\n"%(key, val))
		elif cname=='ndarray' or cname=='matrix':
			mfile.write("%s;\n"%_get_matrix(key, val))
		else:
			mfile.write("%s = %s;\n"%(key, val))
	mfile.close()

	return True


def clean_dir_outdata ():
	"""
	Remove all old testdata files.

	@return Success of operation
	"""

	success=True

	for dname in os.listdir(DIR_OUTPUT):
		if dname=='.svn' or not os.path.isdir(DIR_OUTPUT+os.sep+dname):
			continue

		for fname in os.listdir(DIR_OUTPUT+os.sep+dname):
			if not fname.endswith(EXT_OUTPUT):
				continue

			target=DIR_OUTPUT+os.sep+dname+os.sep+fname
			if os.path.exists(target):
				os.remove(target)
				# os.remove returns False on removal???
				#print 'Could not remove file "%s"'%target
				#success=False

	return success


def get_output (catID, params, prefix=''):
	"""
	Return output data to be written into the testcase's file.

	After computations, the gathered data is structured and
	put into one data structure which can conveniently be written to a
	file that will represent the testcase.

	@param catID ID of entity's category, e.g. category.DISTANCE
	@param params hash with parameters to entity
	@param prefix prefix for parameter's name, e.g. 'subkernel'
	@return Dict containing testcase data ready to be written to file
	"""

	out={}
	prefix=category.get_as_string(catID)+'_'+prefix
	if catID==category.CLASSIFIER:
		out=_get_output_classifier(params, prefix)
	elif catID==category.CLUSTERING:
		out=_get_output_distribution(params, prefix)
	elif catID==category.DISTANCE:
		out=_get_output_distance(params, prefix)
	elif catID==category.DISTRIBUTION:
		out=_get_output_distribution(params, prefix)
	elif catID==category.KERNEL:
		out=_get_output_kernel(params, prefix)
	elif catID==category.PREPROC:
		out=_get_output_preproc(params, prefix)
	elif catID==category.REGRESSION:
		out=_get_output_regression(params, prefix)
	else:
		return out

	out[prefix+'name']=params['name']
	if params.has_key('accuracy'):
		out[prefix+'accuracy']=params['accuracy']

	# example data
	if params.has_key('data'):
		out[prefix+'data_train']=matrix(params['data']['train'])
		out[prefix+'data_test']=matrix(params['data']['test'])


	# params wrt feature class & type
	if params.has_key('feature_class'):
		fclass=params['feature_class']
		ftype=params['feature_type']
		out[prefix+'feature_class']=fclass
		out[prefix+'feature_type']=ftype

		if fclass=='string' or (fclass=='simple' and ftype=='Char'):
			if params.has_key('alphabet'):
				out[prefix+'alphabet']=params['alphabet']
			else:
				out[prefix+'alphabet']='DNA'
			out[prefix+'seqlen']=dataop.LEN_SEQ

		elif fclass=='simple' and ftype=='Byte':
			out[prefix+'alphabet']='RAWBYTE'
			out[prefix+'seqlen']=dataop.LEN_SEQ

		elif fclass=='string_complex':
			if params.has_key('alphabet'):
				out[prefix+'alphabet']=params['alphabet']
			else:
				out[prefix+'alphabet']='DNA'
			if params.has_key('order'):
				out[prefix+'order']=params['order']
			else:
				out[prefix+'order']=featop.WORDSTRING_ORDER
			if params.has_key('gap'):
				out[prefix+'gap']=params['gap']
			else:
				out[prefix+'gap']=featop.WORDSTRING_GAP
			if params.has_key('reverse'):
				out[prefix+'reverse']=params['reverse']
			else:
				out[prefix+'reverse']=featop.WORDSTRING_REVERSE
			if params.has_key('seqlen'):
				out[prefix+'seqlen']=params['seqlen']
			else:
				out[prefix+'seqlen']=dataop.LEN_SEQ

	out['init_random']=dataop.INIT_RANDOM

	return out
