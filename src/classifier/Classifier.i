%define DOCSTR
"The `Classifier` module gathers all classifiers available in the SHOGUN toolkit."
%enddef

%module(docstring=DOCSTR, directors="1") Classifier
%{
 #define SWIG_FILE_WITH_INIT
 #include "features/Labels.h" 
 #include "classifier/Classifier.h" 
%}

#ifdef HAVE_PYTHON
%init %{
   import_array();
%}
#endif

%newobject CClassifier::classify(CLabels* output);
%feature("director");
%feature("autodoc","1");
%rename(Classifier) CClassifier;

%include "lib/common.i"
%include "lib/ShogunException.i"
%include "lib/io.i"
%include "base/Version.i"
%include "base/Parallel.i"
%include "base/SGObject.i"
%include "classifier/Classifier.h" 
%include "classifier/LinearClassifier.i"
%include "classifier/SparseLinearClassifier.i"
%include "classifier/PluginEstimate.i"
%include "distance/DistanceMachine.i" 

/* classifiers based on kernelmachine */
%include "kernel/KernelMachine.i" 
%include "classifier/KernelPerceptron.i"
%include "classifier/svm/SVM.i"
%include "classifier/svm/MultiClassSVM.i"
%include "classifier/svm/LibSVM.i"
%include "classifier/svm/LibSVM_oneclass.i"
%include "classifier/svm/GPBTSVM.i"
%include "classifier/svm/GNPPSVM.i"
