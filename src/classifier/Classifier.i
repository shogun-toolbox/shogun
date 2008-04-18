%define DOCSTR
"The `Classifier` module gathers all classifiers available in the SHOGUN toolkit."
%enddef

%module(docstring=DOCSTR) Classifier
%{
 #define SWIG_FILE_WITH_INIT
 #include "features/Labels.h" 
 #include "classifier/Classifier.h" 
%}

#ifdef HAVE_DOXYGEN
%include "classifier/Classifier_doxygen.i"
#endif

#ifdef HAVE_PYTHON
%init %{
   import_array();
%}
#endif

%newobject CClassifier::classify(CLabels* output);
%feature("autodoc","0");
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

/* classifiers based on kernelmachine */
%include "kernel/KernelMachine.i" 
%include "classifier/KernelPerceptron.i"
%include "classifier/svm/SVM.i"
%include "classifier/svm/MultiClassSVM.i"
%include "classifier/svm/LibSVM.i"
%include "classifier/svm/LibSVMOneClass.i"
%include "classifier/svm/GPBTSVM.i"
%include "classifier/svm/GNPPSVM.i"
%include "classifier/svm/MPDSVM.i"

/* classifiers based on distancemachine */
%include "distance/DistanceMachine.i" 
%include "classifier/KNN.i"

/* classifiers using strings directly */
%include "classifier/svm/WDSVMOcas.i"
