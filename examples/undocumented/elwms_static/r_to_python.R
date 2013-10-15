#demonstrates how to exec python from inside R in the
#eierlegendewollmilchsau shogun interface

library(elwms)
#dyn.load('elwms.so')
#elwms <- function(...) .External("elwms",...,PACKAGE="elwms")

A=matrix(c(1.0,2,3, 4,5,6), nrow = 2, ncol=3)
B=matrix(c(1.0,1,1, 0,0,0), nrow = 2, ncol=3)
pythoncode=sprintf('import numpy\nresults=tuple([A+B])');
elwms('run_python', 'pythoncode', 'print "hi"')
C=elwms('run_python', 'A',A, 'B',B, 'pythoncode', pythoncode)
D=elwms('run_python', 'A',A+1, 'B',B*2, 'pythoncode', pythoncode)
pythoncode=sprintf('import numpy\nresults=(A, B, [ "bla1", "bla2" ])\n');
X=elwms('run_python', 'A',A, 'B',B, 'pythoncode', pythoncode)
print(A)
print(B)
print(C)
print(D)
print(X)

