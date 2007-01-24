%define REGRESSION_DOCSTR
"The `Regression` module gathers all regression methods available in the SHOGUN toolkit."
%enddef

%module(docstring=REGRESSION_DOCSTR, directors="1") Regression
%{
 #define SWIG_FILE_WITH_INIT
%}

#ifdef HAVE_PYTHON
%init %{
   import_array();
%}
#endif

%feature("director");
%feature("autodoc","1");

%include "lib/common.i"
%include "lib/io.i" 
%include "base/Version.i"
%include "base/Parallel.i"
%include "base/SGObject.i"
%include "kernel/KernelMachine.i" 

%include "regression/svr/LibSVR.i"
#ifdef USE_SVMLIGHT
%include "regression/svr/SVR_light.i"
#endif //USE_SVMLIGHT
%include "regression/KRR.i"
