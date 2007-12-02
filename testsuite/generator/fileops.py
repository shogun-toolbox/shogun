import os

import featops
import dataops

DIR_OUTPUT='data'
EXT_OUTPUT='.m'
SVM='svm_'
TYPE='Kernel'

def _get_matrix (name, km):
	line=list()
	lis=list()

	try:
		for x in range(km.shape[0]):
			for y in range(km.shape[1]):
				if(isinstance(km[x,y],(int, long, float, complex))):
					line.append('%.9g' %km[x,y])
				else:
					line.append("'%s'" %km[x,y])
			lis.append(', '.join(line))
			line=list()
	except IndexError:
		for x in range(km.shape[0]):
			if(isinstance(km[x],(int, long, float, complex))):
				line.append('%.9g' %km[x])
			else:
				line.append("'%s'" %km[x])
		lis.append(', '.join(line))
		line=list()
	kmstr=';'.join(lis)
	kmstr=''.join([name, ' = [', kmstr, ']'])

	return kmstr.replace('\n', '')

def _get_filename (output):
	params=[]

	for k, v in output[1].iteritems():
		if (k.find('feature_')!=-1 or k.find('accuracy')!=-1 or
			k.find('data_')!=-1):
			continue
		cn=v.__class__.__name__
		if cn=='bool' or cn=='float' or cn=='int' or cn=='str':
			params.append(str(v))

	params='_'.join(params).replace('.', '')
	if len(params)>0:
		params='_'+params

	return DIR_OUTPUT+os.sep+TYPE.lower()+'s'+os.sep+output[0]+params+EXT_OUTPUT

def write (output):
	if output is None:
		return None

	print 'Writing for '+TYPE+':', output[0]

	mfile=open(_get_filename(output), mode='w')
	mfile.write("name = '"+output[0]+"'\n")

	for k,v in output[1].iteritems():
		cn=v.__class__.__name__
		if cn=='bool' or cn=='str':
			mfile.write("%s = '%s'\n"%(k, v))
		elif cn=='ndarray' or cn=='matrix':
			mfile.write("%s\n"%_get_matrix(k, v))
		else:
			mfile.write("%s = %s\n"%(k, v))

	mfile.close()
	return True

def clean_dir_output ():
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
def get_output_params (name, args=[], prefix='', offset=0):
	if TYPE=='Kernel':
		from klist import KLIST
		data=KLIST[name]
		param='kparam'
	elif TYPE=='Distance':
		from dlist import DLIST
		data=DLIST[name]
		param='dparam'
	else:
		return None

	output={}

	# general params
	output[prefix+'data_class']=data[0][0]
	output[prefix+'data_type']=data[0][1]
	output[prefix+'feature_class']=data[1][0]
	output[prefix+'feature_type']=data[1][1]
	output[prefix+'accuracy']=data[3]

	# params wrt feature class & type
	if data[1][0]=='string' or (data[1][0]=='simple' and data[1][1]=='Char'):
		output[prefix+'alphabet']='DNA'
		output[prefix+'seqlen']=dataops.LEN_SEQ
	elif data[1][0]=='simple' and data[1][1]=='Byte':
		output[prefix+'alphabet']='RAWBYTE'
		output[prefix+'seqlen']=dataops.LEN_SEQ
	elif data[1][0]=='string_complex':
		output[prefix+'order']=featops.WORDSTRING_ORDER
		output[prefix+'gap']=featops.WORDSTRING_GAP
		output[prefix+'reverse']=featops.WORDSTRING_REVERSE
		output[prefix+'alphabet']='DNA'
		output[prefix+'seqlen']=dataops.LEN_SEQ
		output[prefix+'feature_obtain']=data[1][2]

	# arguments, if any
	for i in range(0, len(args)):
		try:
			pname=prefix+param+str(i+offset)+'_'+data[2][i]
		except IndexError:
			break

		cn=args[i].__class__.__name__
		if cn=='int' or cn=='float' or cn=='bool':
			output[pname]=args[i]
		else:
			output[pname]=cn

	return output;
