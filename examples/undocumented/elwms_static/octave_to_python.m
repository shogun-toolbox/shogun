%demonstrates how to exec python from inside octave in the
%eierlegendewollmilchsau shogun interface

A=[[1,2,3];[4,5,6]];
B=[[1,1,1];[0,0,0]];
pythoncode=sprintf('import numpy\nresults=tuple([A+B])\n');
elwms('run_python', 'pythoncode', 'print "hi"')
elwms('run_python', 'pythoncode', 'print "hi2"')
C=elwms('run_python', 'A',A, 'B',B, 'pythoncode', pythoncode)
D=elwms('run_python', 'A',A+1, 'B',B*2, 'pythoncode', pythoncode)
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
