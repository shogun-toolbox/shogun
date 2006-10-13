%{
    #include "kernel/GaussianKernel.h" 
%}

%rename(GaussianKernel) CGaussianKernel;

%include "kernel/SimpleKernel.i" 
%include "kernel/GaussianKernel.h"

