#demonstrates how to exec python from inside R in the
#eierlegendewollmilchsau shogun interface

dyn.load('sg.so')
sg <- function(...) .External("sg",...,PACKAGE="sg")

A=matrix(c(1.0,2,3, 4,5,6), nrow = 2, ncol=3)
B=matrix(c(1.0,1,1, 0,0,0), nrow = 2, ncol=3)
pythoncode=sprintf('import numpy\nresults=dict()\nresults["C"]=A+B');
sg('run_python', 'pythoncode', 'print "hi"')
C=sg('run_python', 'A',A, 'B',B, 'pythoncode', pythoncode)
D=sg('run_python', 'A',A+1, 'B',B*2, 'pythoncode', pythoncode)

print(A)
print(B)
print(C)
print(D)
        
