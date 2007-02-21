from numpy import*

#writes matrix km into the opened m-file fileobj
def print_mat(km, fileobj, mat_name='km'):
	line= list()
	lis = list()
	for x in range(km.shape[0]):
		for y in range(km.shape[1]):
			line.append('%.9g' %km[x,y])	
		lis.append(', '.join(line))
		line = list()
	kmstr = ';'.join(lis)
	kmstr = ''.join([mat_name,' = [',kmstr, ']'])
	kmstr = kmstr.replace('\n','')
	fileobj.write(kmstr)
	fileobj.write('\n')
		
def read_mat(line):
	str   = (line.split('[')[1]).split(']')[0]
	lines = str.split(';')
	lis   = list()
	lis2d = list()
	for x in lines:
		for y in x.split(','):
			lis.append(float(y))
		lis2d.append(lis)
		lis = list()
		
	return array(lis2d)

	#for x in range(km.shape[0]):
	#	line = repr(km[x])
	#	line = line.strip()
	#	remove '...[' and ']...'
	#	line = (line.split('[')[1]).split(']')[0]
	#	lis.append(line)		
	#kmstr = ';'.join(lis)
	#kmstr = ''.join([mat_name,' = [',kmstr, ']'])