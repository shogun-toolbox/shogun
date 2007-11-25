%define REGRESSION_DOCSTR
"The `Regression` module gathers all regression methods available in the SHOGUN toolkit."
%enddef

%module(docstring=REGRESSION_DOCSTR, directors="1") Regression
%{
 #define SWIG_FILE_WITH_INIT
%}

#ifdef USE_DOXYGEN
%include "regression/Regression_doxygen.i"
#endif

#ifdef HAVE_PYTHON
%init %{
   import_array();
%}
#endif

%feature("director");
%feature("autodoc","1");

%include "lib/common.i"
%include "lib/ShogunException.i"
%include "lib/io.i" 
%include "base/Version.i"
%include "base/Parallel.i"
%include "base/SGObject.i"

/* regressors based on kernelmachine */
%include "kernel/KernelMachine.i" 
%include "regression/KRR.i"
%include "classifier/svm/SVM.i"
%include "classifier/svm/LibSVM.i"
%include "regression/svr/LibSVR.i"
#ifdef USE_SVMLIGHT
%include "classifier/svm/SVM_light.i"
%include "regression/svr/SVR_light.i"
#endif //USE_SVMLIGHT
