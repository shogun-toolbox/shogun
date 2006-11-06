%define DOCSTR
"The `Kernel` module gathers all kernels available in the SHOGUN toolkit."
%enddef

%module(docstring=DOCSTR,directors="1") Kernel
%{
#define SWIG_FILE_WITH_INIT
#include "kernel/Kernel.h" 
%}

%include "lib/common.i"

%init %{
	  import_array();
%}

%include "lib/numpy.i"
%apply (DREAL** ARGOUT2, INT* DIM1, INT* DIM2) {(DREAL** dst, INT* m, INT* n)};

%feature("director") CKernel;
%feature("autodoc","0");


%include "kernel/Kernel.h"
%include "kernel/AUCKernel.i"
%include "kernel/PolyKernel.i"
%include "kernel/LinearKernel.i"
%include "kernel/GaussianKernel.i"
%include "kernel/WeightedDegreeCharKernel.i" 
%include "kernel/ConstKernel.i" 
%include "kernel/StringKernel.i" 
/* %include "kernel/SpectrumKernel.i" */
