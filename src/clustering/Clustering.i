%define DOCSTR
"The `Clustering` module gathers all clustering methods available in the SHOGUN toolkit."
%enddef

%module(docstring=DOCSTR,directors="1") Clustering
%{
#define SWIG_FILE_WITH_INIT
%}

%include "lib/common.i"

#ifdef HAVE_DOXYGEN
%include "clustering/Clustering_doxygen.i"
#endif

#ifdef HAVE_PYTHON
%init %{
	  import_array();
%}
#endif

%include "lib/swig_typemaps.i"

%feature("autodoc","0");

%include "lib/ShogunException.i"
%include "lib/io.i"
%include "base/Version.i"
%include "base/Parallel.i"
%include "base/SGObject.i"

%include "classifier/Classifier.h" 
%include "distance/DistanceMachine.i" 
%include "clustering/KMeans.i"
%include "clustering/Hierarchical.i"
