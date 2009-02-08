%{
 #include <shogun/lib/Signal.h>
%}

%rename(Signal) CSignal;
%ignore NUMTRAPPEDSIGS;

%include <shogun/lib/Signal.h>
