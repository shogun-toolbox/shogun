%{
 #include "kernel/CanberraWordKernel.h" 
%}

%rename(CanberraWordKernel) CCanberraWordKernel;

%include "kernel/StringKernel.i" 
%include "kernel/CanberraWordKernel.h" 
