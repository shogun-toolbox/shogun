%define DOCSTR
"The `Distance` module gathers all distances available in the SHOGUN toolkit."
%enddef

%module(docstring=DOCSTR) Distance
%{
#define SWIG_FILE_WITH_INIT
#include "distance/Distance.h"
%}

%include "lib/common.i"
%include "lib/swig_typemaps.i"

#ifdef HAVE_DOXYGEN
%include "distance/Distance_doxygen.i"
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

%include "lib/ShogunException.i"
%include "lib/io.i"
%include "base/Version.i"
%include "base/Parallel.i"
%include "base/SGObject.i"
%include "distance/Distance.h"

%include "distance/SimpleDistance.i"
%include "distance/SparseDistance.i"
%include "distance/RealDistance.i"
%include "distance/StringDistance.i"
%include "distance/CanberraMetric.i"
%include "distance/ChebyshewMetric.i"
%include "distance/GeodesicMetric.i"
%include "distance/JensenMetric.i"
%include "distance/ManhattanMetric.i"
%include "distance/MinkowskiMetric.i"
%include "distance/HammingWordDistance.i"
%include "distance/ManhattanWordDistance.i"
%include "distance/CanberraWordDistance.i"
%include "distance/EuclidianDistance.i"
%include "distance/SparseEuclidianDistance.i"
%include "distance/BrayCurtisDistance.i"
%include "distance/ChiSquareDistance.i"
%include "distance/CosineDistance.i"
%include "distance/TanimotoDistance.i"
