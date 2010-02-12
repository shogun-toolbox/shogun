%demonstrates how to exec python from inside octave in the
%eierlegendewollmilchsau shogun interface

A=[[1,2,3];[4,5,6]];
B=[[1,1,1];[0,0,0]];
%elwms('loglevel', 'ALL');
[s, x,y]=elwms('run_r', 'A', A, 'B', B, 'rfile', 'foo.R')
