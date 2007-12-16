"""
Common operations related to file handling
"""

import os

import featop
import dataop
import config

DIR_OUTPUT='data'
EXT_OUTPUT='.m'

def _get_str_category (category):
	table={
		config.C_KERNEL:'kernel',
		config.C_DISTANCE:'distance',
		config.C_CLASSIFIER:'classifier',
		config.C_CLUSTERING:'clustering',
		config.C_DISTRIBUTION:'distribution',
		config.C_REGRESSION:'regression',
	}

	try:
		return table[category]
	except IndexError:
		return ''

def _get_matrix (name, kmatrix):
	line=list()
	lis=list()

	try:
		for x in range(kmatrix.shape[0]):
			for y in range(kmatrix.shape[1]):
				if (isinstance(kmatrix[x, y], (int, long, float, complex))):
					line.append('%.9g' %kmatrix[x, y])
				else:
					line.append("'%s'" %kmatrix[x, y])
			lis.append(', '.join(line))
			line=list()
	except IndexError:
		for x in range(kmatrix.shape[0]):
			if (isinstance(kmatrix[x], (int, long, float, complex))):
				line.append('%.9g' %kmatrix[x])
			else:
				line.append("'%s'" %kmatrix[x])
		lis.append(', '.join(line))
		line=list()
	str_kmatrix=';'.join(lis)
	str_kmatrix=''.join([name, ' = [', str_kmatrix, ']'])

	return str_kmatrix.replace('\n', '')

def _is_excluded_from_filename (key):
	if (key.find('feature_')!=-1 or
		key.find('accuracy')!=-1 or
		key.find('data_')!=-1 or
		key=='name' or
		key=='regression_type' or
		key=='regression_bias' or
		key=='classifier_bias' or
		key=='classifier_type'):
		return True
	else:
		return False

def _get_filename (category, outdata):
	params=[]

	for key, val in outdata.iteritems():
		if _is_excluded_from_filename(key):
			continue
		cname=val.__class__.__name__
		if cname=='bool' or cname=='float' or cname=='int' or cname=='str':
			params.append(str(val))

	params='_'.join(params).replace('.', '')
	if len(params)>0:
		params='_'+params

	return DIR_OUTPUT+os.sep+_get_str_category(category)+os.sep+ \
		outdata['name']+params+EXT_OUTPUT

############################################################################
# public
############################################################################

def write (category, outdata):
	fnam=_get_filename(category, outdata)
	print 'Writing for '+_get_str_category(category).upper()+': '+ \
		os.path.basename(fnam)

	mfile=open(fnam, mode='w')
	for key, val in outdata.iteritems():
		cname=val.__class__.__name__
		if cname=='bool' or cname=='str':
			mfile.write("%s = '%s'\n"%(key, val))
		elif cname=='ndarray' or cname=='matrix':
			mfile.write("%s\n"%_get_matrix(key, val))
		else:
			mfile.write("%s = %s\n"%(key, val))
	mfile.close()

	return True

def clean_dir_outdata ():
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

# prefix and offset are necessary for subkernels
def get_outdata_params (name, category, args=[], prefix='', offset=0):
	if category==config.C_KERNEL:
		data=config.KERNEL[name]
		argstr='kernel_arg'
	elif category==config.C_DISTANCE:
		data=config.DISTANCE[name]
		argstr='distance_arg'
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

	# arguments, if any
	for i in xrange(len(args)):
		try:
			argname=prefix+argstr+str(i+offset)+'_'+data[2][i]
		except IndexError:
			break

		cname=args[i].__class__.__name__
		if cname=='int' or cname=='float' or cname=='bool' or cname=='str':
			outdata[argname]=args[i]
		else:
			outdata[argname]=cname

	return outdata
