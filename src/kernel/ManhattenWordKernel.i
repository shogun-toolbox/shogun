%{
 #include "kernel/ManhattenWordKernel.h" 
%}

%rename(ManhattenWordKernel) CManhattenWordKernel;

%include "kernel/StringKernel.i" 
%include "kernel/ManhattenWordKernel.h" 
