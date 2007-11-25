%define DOCSTR
"The `Kernel` module gathers all clustering methods available in the SHOGUN toolkit."
%enddef

%module(docstring=DOCSTR,directors="1") Clustering
%{
#define SWIG_FILE_WITH_INIT
%}

#ifdef USE_DOXYGEN
%include "classifier/Classifier_doxygen.i"
#endif

#ifdef HAVE_PYTHON
%init %{
	  import_array();
%}

%include "lib/common.i"
%include "lib/python_typemaps.i"
#endif

/*%feature("director") CKernel;*/
%feature("autodoc","1");

%include "lib/ShogunException.i"
%include "lib/io.i"
%include "base/Version.i"
%include "base/Parallel.i"
%include "base/SGObject.i"
%include "clustering/KMeans.i"
%include "clustering/Hierarchical.i"
