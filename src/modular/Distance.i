%define DOCSTR
"The `Distance` module gathers all distances available in the SHOGUN toolkit."
%enddef

%module(docstring=DOCSTR) Distance
%{
#define SWIG_FILE_WITH_INIT
#include <shogun/distance/Distance.h>
%}

%include "common.i"
%include "swig_typemaps.i"

#ifdef HAVE_DOXYGEN
%include "Distance_doxygen.i"
#endif

#ifdef HAVE_PYTHON
%init %{
	  import_array();
%}
%feature("autodoc", "get_distance_matrix(self) -> numpy 2dim array of float") get_distance_matrix;
#endif

%apply (float64_t** ARGOUT2, int32_t* DIM1, int32_t* DIM2) {(float64_t** dst, int32_t* m, int32_t* n)};

%rename(Distance) CDistance;
%feature("autodoc","0");

%include "ShogunException.i"
%include "io.i"
%include "Version.i"
%include "Parallel.i"
%include "SGObject.i"
%include <shogun/distance/Distance.h>

%include "SimpleDistance.i"
%include "SparseDistance.i"
%include "RealDistance.i"
%include "StringDistance.i"
%include "CanberraMetric.i"
%include "ChebyshewMetric.i"
%include "GeodesicMetric.i"
%include "JensenMetric.i"
%include "ManhattanMetric.i"
%include "MinkowskiMetric.i"
%include "HammingWordDistance.i"
%include "ManhattanWordDistance.i"
%include "CanberraWordDistance.i"
%include "EuclidianDistance.i"
%include "SparseEuclidianDistance.i"
%include "BrayCurtisDistance.i"
%include "ChiSquareDistance.i"
%include "CosineDistance.i"
%include "TanimotoDistance.i"
