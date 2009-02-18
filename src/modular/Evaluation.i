%define DOCSTR
"The `Evaluation` module is a collection of classes like PerformanceMeasures for the SHOGUN toolkit."
%enddef

%module(docstring=DOCSTR) Evaluation
%{
#define SWIG_FILE_WITH_INIT
%}

%include "init.i"
%include "common.i"
%include "swig_typemaps.i"

#ifdef HAVE_DOXYGEN
%include "Evaluation_doxygen.i"
#endif

#ifdef HAVE_PYTHON
%init %{
	  import_array();
%}
#endif

%feature("autodoc","0");

%include "ShogunException.i"
%include "io.i"
%include "SGObject.i"

%include "PerformanceMeasures.i"
