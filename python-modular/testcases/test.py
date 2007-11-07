#!/usr/bin/env python

import test_kernels
from numpy import *
import sys
from os import listdir


def get_name_fun (name):
	prefix_svm='svm_'
	prefix=''

	if (file.find(prefix_svm)>-1):
		prefix=prefix_svm

	return 'test_kernels.'+prefix+name[:-len('Kernel')].lower()

def test_mfile (file):
	mfile=open(file, mode='r')
	input={}

	for line in mfile:
		param = line.split('=')[0].strip()
		
		if param=='name':
			name=line.split('=')[1].strip().split("'")[1]
			name_fun=get_name_fun(name)
			input[param]=name
		elif param=='km_train':
			input['km_train']=read_matrix(line)
		elif param=='km_test':
			input['km_test']=read_matrix(line)
		elif param=='data_train':
			input['data_train']=read_matrix(line)
		elif param=='data_test':
			input['data_test']=read_matrix(line)
		else:
			if (line.find("'")==-1):
				input[param]=eval(line.split('=')[1])
			else: 
				input[param]=line.split('=')[1].strip().split("'")[1]

	mfile.close()
	fun=eval(name_fun)

	return fun(input)

def read_matrix (line):
	str=(line.split('[')[1]).split(']')[0]
	lines=str.split(';')
	lis2d=list()

	for x in lines:
		lis=list()
		for y in x.split(','):
			y=y.replace("'","").strip()
			if(y.isalpha()):
				lis.append(y)
			else:
				lis.append(float(y))
		lis2d.append(lis)

	return array(lis2d)

for file in sys.argv:
	if (file.endswith('.m')):
		try:
			res=test_mfile(file)
			if res:
				sys.exit(0)
			else:
				sys.exit(1)
		except KeyError:
			print 'Error in input test data'
			sys.exit(2)
