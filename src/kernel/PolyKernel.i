%{
 #include "kernel/PolyKernel.h" 
%}

%rename(PolyKernel) CPolyKernel;

%include "kernel/SimpleKernel.i" 
%include "kernel/PolyKernel.h" 
