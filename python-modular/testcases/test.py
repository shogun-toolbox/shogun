import test_kernels
import read_mfile
import sys
from os import listdir


def test_mfile(file):
	mfile = open(file, mode='r')

	param_dict = {}

	for line in mfile:
		parname = line.split('=')[0].strip()
		
		if parname=='functionname':
			name =  line.split('=')[1].strip().split("'")[1]
			functionname = 'test_kernels.'+name

		elif parname=='km_train':
			param_dict['km_train'] = read_mfile.read_mat(line)

		elif parname=='km_test':
			param_dict['km_test'] = read_mfile.read_mat(line)	

		elif parname=='traindat':
			param_dict['traindat'] = read_mfile.read_mat(line)

		elif parname=='testdat':
			param_dict['testdat'] =  read_mfile.read_mat(line)
		else :
			if(line.find("'")==-1):
				param_dict[parname]= eval(line.split('=')[1])
			else: 
				param_dict[parname]= line.split('=')[1].strip().split("'")[1]
	

	mfile.close()

	test_fun = eval(functionname)

	return test_fun(param_dict )


mfiles = sys.argv
for file in mfiles:
	if(file.endswith('.m',0,sys.maxint)):
		res = test_mfile(file)
		if res:
			sys.exit(0)
		else:
			sys.exit(1)


sys.exit(1)