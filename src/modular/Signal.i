%{
 #include "lib/Signal.h" 
%}

%rename(Signal) CSignal;
%ignore NUMTRAPPEDSIGS;

%include "lib/Signal.h" 
