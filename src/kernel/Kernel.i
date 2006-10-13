%module(directors="1") Kernel
%{
#include "kernel/Kernel.h" 
%}

%include "lib/common.i"

%feature("director") CKernel;

%include "kernel/Kernel.h"
%include "kernel/GaussianKernel.i"
%include "kernel/WeightedDegreeCharKernel.i" 
