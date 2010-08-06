%define DOCSTR
"The `Shogun` module gathers everything available in the SHOGUN toolkit."
%enddef


%module(docstring=DOCSTR) Shogun
%{
 #define SWIG_FILE_WITH_INIT
 #include "features/Labels.h" 
 #include "features/Features.h" 
 #include "kernel/Kernel.h"
 #include "classifier/Classifier.h"
 #include "distributions/Distribution.h"

%}


#ifdef HAVE_PYTHON
%init %{
   import_array();
%}
#endif

%include "std_string.i"
%include "Classifier.i"
%include "Clustering.i"
%include "Distance.i"
%include "Distribution.i"
%include "Evaluation.i"
%include "Features.i"
%include "Kernel.i"
%include "Library.i"
%include "PreProc.i"
%include "Regression.i"
%include "SGBase.i"
%include "Shogun.i"
%include "Structure.i"
