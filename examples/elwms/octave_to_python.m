%demonstrates how to exec python from inside octave in the
%eierlegendewollmilchsau shogun interface

A=[[1,2,3];[4,5,6]];
B=[[1,1,1];[0,0,0]];
pythoncode=sprintf('import numpy\nresults=dict()\nresults["C"]=A+B');
C=sg('run_python', 'A',A, 'B',B, 'pythoncode', pythoncode)
D=sg('run_python', 'A',A+1, 'B',B*2, 'pythoncode', pythoncode)
sg('run_python', 'pythoncode', 'print "hi"')
