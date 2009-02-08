%define DOCSTR
"The `Kernel` module gathers all kernels available in the SHOGUN toolkit."
%enddef

%module(docstring=DOCSTR) Kernel
%{
#define SWIG_FILE_WITH_INIT
#include <shogun/kernel/Kernel.h>
%}

#ifdef HAVE_DOXYGEN
%include "Kernel_doxygen.i"
#endif

%include "common.i"
%include "swig_typemaps.i"

#ifdef HAVE_PYTHON
%init %{
	  import_array();
%}
%feature("autodoc", "get_kernel_matrix(self) -> numpy 2dim array of float") get_kernel_matrix;
#endif

%apply (float64_t** ARGOUT2, int32_t* DIM1, int32_t* DIM2) {(float64_t** dst, int32_t* m, int32_t* n)};

%rename(Kernel) CKernel;
%feature("autodoc","0");

%include "ShogunException.i"
%include "io.i"
%include "Version.i"
%include "Parallel.i"
%include "SGObject.i"
%include <shogun/kernel/Kernel.h>

%include "StringKernel.i" 
%include "SimpleKernel.i"
%include "SparseKernel.i"

%include "ConstKernel.i" 
%include "DiagKernel.i" 
%include "LinearKernel.i"
%include "PolyKernel.i"
%include "GaussianKernel.i"
%include "GaussianShortRealKernel.i"
%include "GaussianShiftKernel.i"
%include "SigmoidKernel.i"
%include "SparseGaussianKernel.i"
%include "SparseLinearKernel.i"
%include "SparsePolyKernel.i"
%include "TensorProductPairKernel.i"

%include "Chi2Kernel.i"
%include "AUCKernel.i"

%include "WeightedDegreeStringKernel.i" 
%include "WeightedDegreePositionStringKernel.i" 
%include "CommUlongStringKernel.i" 
%include "CommWordStringKernel.i" 
%include "WeightedCommWordStringKernel.i" 
%include "MindyGramKernel.i"
%include "OligoKernel.i"
%include "FixedDegreeStringKernel.i"
%include "HistogramWordStringKernel.i"
%include "LinearByteKernel.i"
%include "LinearStringKernel.i"
%include "LocalAlignmentStringKernel.i"
%include "LinearWordKernel.i"

%include "LocalityImprovedStringKernel.i"
%include "PolyMatchStringKernel.i"
%include "PolyMatchWordStringKernel.i"
%include "SalzbergWordStringKernel.i"
%include "SimpleLocalityImprovedStringKernel.i"
%include "MatchWordStringKernel.i"

%include "CombinedKernel.i"
%include "CustomKernel.i"
%include "DistanceKernel.i"

%include "KernelNormalizer.i"
%include "AvgDiagKernelNormalizer.i"
%include "IdentityKernelNormalizer.i"
%include "SqrtDiagKernelNormalizer.i"
