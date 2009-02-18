%define DOCSTR
"The `Classifier` module gathers all classifiers available in the SHOGUN toolkit."
%enddef

%module(docstring=DOCSTR) Classifier
%{
 #define SWIG_FILE_WITH_INIT
 #include <shogun/features/Labels.h>
 #include <shogun/classifier/Classifier.h>
%}

#ifdef HAVE_DOXYGEN
%include "Classifier_doxygen.i"
#endif

#ifdef HAVE_PYTHON
%init %{
   import_array();
%}
#endif

%newobject CClassifier::classify(CLabels* output);
%feature("autodoc","0");
%rename(Classifier) CClassifier;

%include "init.i"
%include "common.i"
%include "ShogunException.i"
%include "io.i"
%include "Version.i"
%include "Parallel.i"
%include "SGObject.i"
%include <shogun/classifier/Classifier.h>
%include "LinearClassifier.i"
%include "PluginEstimate.i"

/* classifiers based on kernelmachine */
%include "KernelMachine.i" 
%include "KernelPerceptron.i"
%include "SVM.i"
%include "MultiClassSVM.i"
%include "LibSVM.i"
%include "LibSVMOneClass.i"
%include "GPBTSVM.i"
%include "GNPPSVM.i"
%include "MPDSVM.i"

/* classifiers based on distancemachine */
%include "DistanceMachine.i" 
%include "KNN.i"

/* classifiers using strings directly */
%include "WDSVMOcas.i"
