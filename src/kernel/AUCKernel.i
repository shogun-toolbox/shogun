%{
 #include "kernel/AUCKernel.h" 
%}

%rename(AUCKernel) CAUCKernel;

%include "kernel/SimpleKernel.i" 
%include "kernel/AUCKernel.h" 
