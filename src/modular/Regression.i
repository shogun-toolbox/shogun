%define REGRESSION_DOCSTR
"The `Regression` module gathers all regression methods available in the SHOGUN toolkit."
%enddef

%module(docstring=REGRESSION_DOCSTR) Regression
%{
 #define SWIG_FILE_WITH_INIT
 #include <shogun/regression/Regression.h>
%}

#ifdef HAVE_DOXYGEN
%include "Regression_doxygen.i"
#endif

#ifdef HAVE_PYTHON
%init %{
   import_array();
%}
#endif

%feature("autodoc","0");

%include "init.i"
%include "common.i"
%include "ShogunException.i"
%include "io.i"
%include "Version.i"
%include "Parallel.i"
%include "SGObject.i"

/* regressors based on kernelmachine */
%include "KernelMachine.i"
%include <shogun/regression/Regression.h>
%include "KRR.i"
%include "SVM.i"
%include "LibSVM.i"
%include "LibSVR.i"
#ifdef USE_SVMLIGHT
%include "SVM_light.i"
%include "SVR_light.i"
#endif //USE_SVMLIGHT
