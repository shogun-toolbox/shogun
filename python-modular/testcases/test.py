import test_kernels
from numpy import *
import sys
from os import listdir


def test_mfile (file):
	mfile=open(file, mode='r')
	params={}

	for line in mfile:
		param = line.split('=')[0].strip()
		
		if param=='functionname':
			name_fun='test_kernels.'+line.split('=')[1].strip().split("'")[1]
		elif param=='km_train':
			params['km_train']=read_matrix(line)
		elif param=='km_test':
			params['km_test']=read_matrix(line)
		elif param=='data_train':
			params['data_train']=read_matrix(line)
		elif param=='data_test':
			params['data_test']=read_matrix(line)
		else :
			if (line.find("'")==-1):
				params[param]=eval(line.split('=')[1])
			else: 
				params[param]=line.split('=')[1].strip().split("'")[1]

	mfile.close()

	test_fun = eval(name_fun)
	return test_fun(params)

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

sys.exit(0)
