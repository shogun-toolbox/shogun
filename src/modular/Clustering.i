%define DOCSTR
"The `Clustering` module gathers all clustering methods available in the SHOGUN toolkit."
%enddef

%module(docstring=DOCSTR) Clustering
%{
#define SWIG_FILE_WITH_INIT
%}

%include "common.i"

#ifdef HAVE_DOXYGEN
%include "clustering/Clustering_doxygen.i"
#endif

#ifdef HAVE_PYTHON
%init %{
	  import_array();
%}
#endif

%include "swig_typemaps.i"

%feature("autodoc","0");

%include "ShogunException.i"
%include "io.i"
%include "Version.i"
%include "Parallel.i"
%include "SGObject.i"

%include <shogun/classifier/Classifier.h> 
%include "DistanceMachine.i" 
%include "KMeans.i"
%include "Hierarchical.i"
