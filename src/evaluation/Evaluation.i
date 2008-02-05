%define DOCSTR
"The `Evaluation` module is a collection of classes like PerformanceMeasures for the SHOGUN toolkit."
%enddef

%module(docstring=DOCSTR,directors="1") Evaluation
%feature("autodoc","0");

#ifdef HAVE_DOXYGEN
%include "evaluation/Evaluation_doxygen.i"
#endif

%include "lib/common.i"
%include "lib/ShogunException.i"
%include "lib/io.i"
%include "base/SGObject.i"

%include "evaluation/PerformanceMeasures.i"
