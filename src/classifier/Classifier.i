%define DOCSTR
"The `Classifier` module gathers all classifiers available in the SHOGUN toolkit."
%enddef

%module(docstring=DOCSTR, directors="1") Classifier
%{
 #define SWIG_FILE_WITH_INIT
 #include "classifier/Classifier.h" 
%}

#ifdef HAVE_PYTHON
%init %{
   import_array();
%}
#endif

%feature("director");
%feature("autodoc","1");
%rename(Classifer) CClassifier;

%include "lib/common.i"
%include "lib/ShogunException.i"
%include "lib/io.i"
%include "base/Version.i"
%include "base/Parallel.i"
%include "base/SGObject.i"
%include "classifier/Classifier.h" 

%include "kernel/KernelMachine.i" 
%include "distance/DistanceMachine.i" 
%include "classifier/KernelPerceptron.i"
%include "classifier/LDA.i"
%include "classifier/LPM.i"
%include "classifier/LinearClassifier.i"
%include "classifier/Perceptron.i"
%include "classifier/PluginEstimate.i"
%include "classifier/KNN.i"
%include "classifier/svm/SVM.i"
%include "classifier/svm/SVMLin.i"
