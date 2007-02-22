from numpy import*

		
def read_mat(line):
	str   = (line.split('[')[1]).split(']')[0]
	lines = str.split(';')
	lis   = list()
	lis2d = list()
	for x in lines:
		for y in x.split(','):
			y = y.replace("'","").strip()
			if(y.isalpha()):
				lis.append(y)
			else:
				lis.append(float(y))
		lis2d.append(lis)
		lis = list()
		
	return array(lis2d)

