%{
 #include "kernel/CommUlongStringKernel.h" 
%}

%rename(CommUlongStringKernel) CCommUlongStringKernel;

%include "kernel/StringKernel.i" 
%include "kernel/CommUlongStringKernel.h" 
