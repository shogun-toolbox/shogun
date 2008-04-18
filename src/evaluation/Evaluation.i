%define DOCSTR
"The `Evaluation` module is a collection of classes like PerformanceMeasures for the SHOGUN toolkit."
%enddef

%module(docstring=DOCSTR) Evaluation
%{
#define SWIG_FILE_WITH_INIT
%}

%include "lib/common.i"
%include "lib/swig_typemaps.i"

#ifdef HAVE_DOXYGEN
%include "evaluation/Evaluation_doxygen.i"
#endif

#ifdef HAVE_PYTHON
%init %{
	  import_array();
%}
#endif

%feature("autodoc","0");

%include "lib/ShogunException.i"
%include "lib/io.i"
%include "base/SGObject.i"

%include "evaluation/PerformanceMeasures.i"
