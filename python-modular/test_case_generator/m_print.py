from numpy import*

#writes matrix km into the opened m-file fileobj
def print_mat(km, fileobj, mat_name='km'):
	line= list()
	lis = list()
	for x in range(km.shape[0]):
		for y in range(km.shape[1]):
			if(isinstance(km[x,y],(int, long, float, complex))):
				line.append('%.9g' %km[x,y])	
			else:
				line.append("'%c'" %km[x,y])
		lis.append(', '.join(line))
		line = list()
	kmstr = ';'.join(lis)
	kmstr = ''.join([mat_name,' = [',kmstr, ']'])
	kmstr = kmstr.replace('\n','')
	fileobj.write(kmstr)
	fileobj.write('\n')

		

