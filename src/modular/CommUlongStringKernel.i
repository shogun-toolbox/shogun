%{
 #include <shogun/kernel/CommUlongStringKernel.h>
%}

%rename(CommUlongStringKernel) CCommUlongStringKernel;

%include "StringKernel.i" 
%include <shogun/kernel/CommUlongStringKernel.h>
