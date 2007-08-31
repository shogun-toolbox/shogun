%{
 #include "kernel/HammingWordKernel.h" 
%}

%rename(HammingWordKernel) CHammingWordKernel;

%include "kernel/StringKernel.i" 
%include "kernel/HammingWordKernel.h" 
