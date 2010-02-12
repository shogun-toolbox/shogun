from elwms import elwms
import numpy
x=numpy.array([[1,2,3],[4,5,6]],dtype=numpy.float64)
y=numpy.array([[7,8,9],[0,1,2]],dtype=numpy.float64)

#elwms('loglevel', 'ALL')
#elwms('run_octave','octavecode', 'disp("hi")')
a,b,c=elwms('run_octave','x', x, 'y', y,
		'octavecode', 'class(x), disp(x), results=list(x+y,1,{"a"})')
res1=elwms('run_octave','x', x, 'y', y,
		'octavecode', 'disp(x); disp(y); results=x+y+rand(2,3)\n')
res2=elwms('run_octave','A', ['test','bla','foo'],
		'octavecode',
'''
disp(A);
disp("hi");
results={"a","b","c"}
''')
