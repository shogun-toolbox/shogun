"""Common operations related to file handling"""

import os

import featop
import dataop
import config
from numpy import ushort, ubyte, double

DIR_OUTPUT='data'
EXT_OUTPUT='.m'


def _get_str_category (category):
	"""Returns the string representation of given category.

	@param category ID of a category
	@return String of the category or empty string if ID was invalid
	"""

	table={
		config.C_KERNEL:'kernel',
		config.C_DISTANCE:'distance',
		config.C_CLASSIFIER:'classifier',
		config.C_CLUSTERING:'clustering',
		config.C_DISTRIBUTION:'distribution',
		config.C_REGRESSION:'regression',
		config.C_PREPROC:'preproc',
	}

	try:
		return table[category]
	except IndexError:
		return ''


def _get_matrix (name, kmatrix):
	"""Converts a numpy matrix into a matrix digestable by e.g. matlab.

	@param name Name of the matrix
	@param kmatrix The matrix
	@return String which contains a matrix digestable by e.g. matlab
	"""

	line=list()
	list_numbers=(int, long, float, complex)
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
	"""Determine if given key's value shall not be part of the filename.

	@param key Name of the value to check for.
	@return True if shall be excluded, false otherwise
	"""

	if (key.find('feature_')!=-1 or
		key.find('accuracy')!=-1 or
		key.find('data_')!=-1 or
		key=='name' or
		key=='init_random' or
		key=='regression_type' or
		key=='regression_bias' or
		key=='distribution_likelihood' or
		key=='distribution_derivatives' or
		key=='distribution_best_path' or
		key=='distribution_best_path_state' or
		key=='classifier_bias' or
		key=='classifier_type'):
		return True
	else:
		return False


def _get_filename (category, outdata):
	"""Return filename for testcase data's output.

	@param category ID of the category
	@param outdata data to be written into file
	@return String with the filename
	"""

	params=[]

	for key, val in outdata.iteritems():
		if _is_excluded_from_filename(key):
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

	return DIR_OUTPUT+os.sep+_get_str_category(category)+os.sep+ \
		outdata['name']+params+EXT_OUTPUT

############################################################################
# public
############################################################################

def write (category, outdata):
	"""Write given testcase data to a file.

	@param category ID of the category, like C_KERNEL
	@param outdata data to be written into file
	@return Success of operation
	"""

	fnam=_get_filename(category, outdata)
	print 'Writing for '+_get_str_category(category).upper()+': '+ \
		os.path.basename(fnam)

	mfile=open(fnam, mode='w')
	for key, val in outdata.iteritems():
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
	"""Remove all old testdata files.

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

def get_args (prefix, names, args, offset=0):
	"""Return argument list for kernel/distance to be written to a file.

	@param prefix Prefix for argument's name, like 'subkernel'
	@param names Names of the arguments
	@param args Values of the arguments
	@param offset Offset from argument conter, used for e.g. subkernels
	@return Dict with data on arguments ready to be written to file
	"""

	outdata={}

	for i in xrange(len(args)):
		try:
			name=prefix+'_arg'+str(i+offset)+'_'+names[i]
		except IndexError:
			break

		cname=args[i].__class__.__name__
		if cname=='int' or cname=='float' or cname=='bool' or cname=='str':
			outdata[name]=args[i]
		else:
			outdata[name]=cname

	return outdata

# prefix and offset are necessary for subkernels
def get_outdata (name, category, args=(), prefix='', offset=0):
	"""Return data to be written into the testcase's file.

	After computations and such, the gathered data is structured and
	put into one data structure which can conveniently be written to a
	file that will represent the testcase.
	
	@param name Kernel/Distance's name
	@param category ID of the category, like C_DISTANCE
	@param args argument list of the item in question
	@param prefix Prefix for argument's name, like 'subkernel'
	@param offset Offset from argument conter, used for e.g. subkernels
	@return Dict containing testcase data ready to be written to file
	"""

	if category==config.C_KERNEL:
		data=config.KERNEL[name]
		prefix_arg=prefix+'kernel'
	elif category==config.C_DISTANCE:
		data=config.DISTANCE[name]
		prefix_arg=prefix+'distance'
	else:
		return {}

	outdata={}

	# general params
	outdata[prefix+'data_class']=data[0][0]
	outdata[prefix+'data_type']=data[0][1]
	outdata[prefix+'feature_class']=data[1][0]
	outdata[prefix+'feature_type']=data[1][1]
	outdata[prefix+'accuracy']=data[3]

	# params wrt feature class & type
	if data[1][0]=='string' or (data[1][0]=='simple' and data[1][1]=='Char'):
		outdata[prefix+'alphabet']='DNA'
		outdata[prefix+'seqlen']=dataop.LEN_SEQ
	elif data[1][0]=='simple' and data[1][1]=='Byte':
		outdata[prefix+'alphabet']='RAWBYTE'
		outdata[prefix+'seqlen']=dataop.LEN_SEQ
	elif data[1][0]=='string_complex':
		outdata[prefix+'order']=featop.WORDSTRING_ORDER
		outdata[prefix+'gap']=featop.WORDSTRING_GAP
		outdata[prefix+'reverse']=featop.WORDSTRING_REVERSE
		outdata[prefix+'alphabet']='DNA'
		outdata[prefix+'seqlen']=dataop.LEN_SEQ
		outdata[prefix+'feature_obtain']=data[1][2]

	if args!=(): # arguments, if any
		outdata.update(get_args(prefix_arg, data[2], args, offset))

	outdata['init_random']=dataop.INIT_RANDOM

	return outdata
