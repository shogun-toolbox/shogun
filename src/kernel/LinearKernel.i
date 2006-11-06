%{
 #include "kernel/LinearKernel.h" 
%}

%rename(LinearKernel) CLinearKernel;

%include "kernel/SimpleKernel.i" 
%include "kernel/LinearKernel.h" 
