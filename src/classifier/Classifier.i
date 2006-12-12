%module(directors="1") Classifier
%{
 #define SWIG_FILE_WITH_INIT
 #include "classifier/Classifier.h" 
%}

%include "lib/common.i"

%init %{
   import_array();
%}

%feature("director");

%rename(Classifer) CClassifier;

%include "classifier/Classifier.h" 

%include "classifier/KernelPerceptron.i"
%include "classifier/LDA.i"
%include "classifier/LPM.i"
%include "classifier/LinearClassifier.i"
%include "classifier/Perceptron.i"
%include "classifier/PluginEstimate.i"
%include "classifier/KNN.i"
%include "classifier/svm/SVM.i"
%include "classifier/svm/SVMLin.i"
