%{
 #include <shogun/kernel/CommWordStringKernel.h>
%}

%rename(CommWordStringKernel) CCommWordStringKernel;

%include "StringKernel.i" 
%include <shogun/kernel/CommWordStringKernel.h>
