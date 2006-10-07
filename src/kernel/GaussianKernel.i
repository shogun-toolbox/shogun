%module GaussianKernel

%include "features/RealFeatures.i"
%include "kernel/RealKernel.i"

%{
    #include "features/RealFeatures.h" 
	#include "kernel/RealKernel.h"
    #include "kernel/GaussianKernel.h" 
%}

%feature("notabstract") CGaussianKernel;
%include "kernel/GaussianKernel.h"
