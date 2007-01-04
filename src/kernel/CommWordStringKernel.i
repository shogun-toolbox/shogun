%{
 #include "kernel/CommWordStringKernel.h" 
%}

%rename(CommWordStringKernel) CCommWordStringKernel;

%include "kernel/StringKernel.i" 
%include "kernel/CommWordStringKernel.h" 
