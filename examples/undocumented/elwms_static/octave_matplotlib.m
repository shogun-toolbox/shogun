% uses matplotlib to plot a figure from within octave

A=[[1,2,3];[4,5,6]];
B=[[1,1,1];[0,0,0]];
pythoncode=sprintf("\
\
import numpy\n\
x=numpy.array([[1.0,2,3],[4,5,6]])\n\
results=(A, B, [ 'bla1', 'bla2' ], x)\n\
\
from pylab import *\n\
plot(B)\n\
show()\n\
\
")

[A2,B2,bla,x]=elwms('run_python', 'A',A, 'B',B, 'pythoncode', pythoncode)
sprintf('%s\n', char(bla{1}))
sprintf('%s\n', char(bla{2}))
