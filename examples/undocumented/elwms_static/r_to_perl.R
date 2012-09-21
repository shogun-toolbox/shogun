#demonstrates how to exec python from inside R in the
#demonstrates how to exec perl from inside R in the
#eierlegendewollmilchsau shogun interface

library(elwms)
#dyn.load('elwms.so')
#elwms <- function(...) .External("elwms",...,PACKAGE="elwms")

A=matrix(c(1.0,2,3, 4,5,6), nrow = 2, ncol=3)
B=matrix(c(1.0,1,1, 0,0,0), nrow = 2, ncol=3)
pythoncode=sprintf('import numpy\nresults=tuple([A+B])');
perlcode=sprintf('import numpy\nresults=tuple([A+B])');
elwms('run_python', 'pythoncode', 'print "hi"')
elwms('run_perl', 'perlcode', 'print "hi"')
C=elwms('run_python', 'A',A, 'B',B, 'pythoncode', pythoncode)
C=elwms('run_perl', 'A',A, 'B',B, 'perlcode', perlcode)
D=elwms('run_python', 'A',A+1, 'B',B*2, 'pythoncode', pythoncode)
D=elwms('run_perl', 'A',A+1, 'B',B*2, 'perlcode', perlcode)
pythoncode=sprintf('import numpy\nresults=(A, B, [ "bla1", "bla2" ])\n');
perlcode=sprintf('import numpy\nresults=(A, B, [ "bla1", "bla2" ])\n');
X=elwms('run_python', 'A',A, 'B',B, 'pythoncode', pythoncode)
X=elwms('run_perl', 'A',A, 'B',B, 'perlcode', perlcode)
print(A)
print(B)
print(C)
print(D)
print(X)
        
