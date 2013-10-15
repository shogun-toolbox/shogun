#demonstrates how to exec python from inside R in the
#eierlegendewollmilchsau shogun interface

library(elwms)
#dyn.load('elwms.so')
#elwms <- function(...) .External("elwms",...,PACKAGE="elwms")

A=matrix(c(1.0,2,3, 4,5,6), nrow = 2, ncol=3)
B=matrix(c(1.0,1,1, 0,0,0), nrow = 2, ncol=3)
octavecode=sprintf('results=A+B');
elwms('run_octave', 'octavecode', 'disp("hi")')
C=elwms('run_octave', 'A',A, 'B',B, 'octavecode', octavecode)
D=elwms('run_octave', 'A',A+1, 'B',B*2, 'octavecode', octavecode)
octavecode=sprintf('results=list(A, B,{"bla1","bla2"})\n');
X=elwms('run_octave', 'A',A, 'B',B, 'octavecode', octavecode)
print(A)
print(B)
print(C)
print(D)
print(X)

