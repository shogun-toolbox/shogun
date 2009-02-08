%{
 #include <shogun/kernel/AUCKernel.h>
%}

%rename(AUCKernel) CAUCKernel;

%include "SimpleKernel.i" 
%include <shogun/kernel/AUCKernel.h> 
