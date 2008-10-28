%define DOCSTR
"The `Kernel` module gathers all kernels available in the SHOGUN toolkit."
%enddef

%module(docstring=DOCSTR) Kernel
%{
#define SWIG_FILE_WITH_INIT
#include "kernel/Kernel.h"
%}

#ifdef HAVE_DOXYGEN
%include "kernel/Kernel_doxygen.i"
#endif

%include "lib/common.i"
%include "lib/swig_typemaps.i"

#ifdef HAVE_PYTHON
%init %{
	  import_array();
%}
%feature("autodoc", "get_kernel_matrix(self) -> numpy 2dim array of float") get_kernel_matrix;
#endif

%apply (float64_t** ARGOUT2, int32_t* DIM1, int32_t* DIM2) {(float64_t** dst, int32_t* m, int32_t* n)};

%rename(Kernel) CKernel;
%feature("autodoc","0");

%include "lib/ShogunException.i"
%include "lib/io.i"
%include "base/Version.i"
%include "base/Parallel.i"
%include "base/SGObject.i"
%include "kernel/Kernel.h"

%include "kernel/StringKernel.i" 
%include "kernel/SimpleKernel.i"
%include "kernel/SparseKernel.i"

%include "kernel/ConstKernel.i" 
%include "kernel/DiagKernel.i" 
%include "kernel/LinearKernel.i"
%include "kernel/PolyKernel.i"
%include "kernel/GaussianKernel.i"
%include "kernel/GaussianShortRealKernel.i"
%include "kernel/GaussianShiftKernel.i"
%include "kernel/SigmoidKernel.i"
%include "kernel/SparseGaussianKernel.i"
%include "kernel/SparseLinearKernel.i"
%include "kernel/SparsePolyKernel.i"

%include "kernel/Chi2Kernel.i"
%include "kernel/AUCKernel.i"

%include "kernel/WeightedDegreeStringKernel.i" 
%include "kernel/WeightedDegreePositionStringKernel.i" 
%include "kernel/CommUlongStringKernel.i" 
%include "kernel/CommWordStringKernel.i" 
%include "kernel/WeightedCommWordStringKernel.i" 
%include "kernel/MindyGramKernel.i"
%include "kernel/OligoKernel.i"
%include "kernel/FixedDegreeStringKernel.i"
%include "kernel/HistogramWordStringKernel.i"
%include "kernel/LinearByteKernel.i"
%include "kernel/LinearStringKernel.i"
%include "kernel/LocalAlignmentStringKernel.i"
%include "kernel/LinearWordKernel.i"

%include "kernel/LocalityImprovedStringKernel.i"
%include "kernel/PolyMatchStringKernel.i"
%include "kernel/PolyMatchWordStringKernel.i"
%include "kernel/SalzbergWordStringKernel.i"
%include "kernel/SimpleLocalityImprovedStringKernel.i"
%include "kernel/MatchWordStringKernel.i"

%include "kernel/CombinedKernel.i"
%include "kernel/CustomKernel.i"
%include "kernel/DistanceKernel.i"

%include "kernel/KernelNormalizer.i"
%include "kernel/AvgDiagKernelNormalizer.i"
%include "kernel/IdentityKernelNormalizer.i"
%include "kernel/SqrtDiagKernelNormalizer.i"
