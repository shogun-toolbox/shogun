from numpy import*

		
def read_mat(line):
	str   = (line.split('[')[1]).split(']')[0]
	lines = str.split(';')
	lis   = list()
	lis2d = list()
	char  = False 
	for x in lines:
		for y in x.split(','):
			y = y.replace("'","").strip()
			if(y.isalpha()):
				lis.append(y)
				char = True
			else:
				lis.append(float(y))
		lis2d.append(lis)
		lis = list()
		
	if char:
		n = size(lis2d)/size(lis2d[0])
		m = size(lis2d[0])
		mat = chararray((n,m),1,order='FORTRAN') 
		print "mat", mat.shape, " n: ",n, " m: ",m
		for i in range(n):
			for j in range(m):
	        		mat[i][j]=lis2d[i][j] 
		return mat
	else: 
		return array(lis2d)
