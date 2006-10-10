%module GaussianKernel

%include "features/RealFeatures.i"
%include "kernel/SimpleKernel.i"

%{
    #include "features/RealFeatures.h" 
	#include "kernel/SimpleKernel.h"
    #include "kernel/GaussianKernel.h" 
%}

%feature("notabstract") CGaussianKernel;
%include "kernel/GaussianKernel.h"
