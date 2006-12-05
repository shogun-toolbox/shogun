%{
 #include "classifier/LinearClassifier.h" 
%}
%rename(LinearClassifier) CLinearClassifier;

%include "classifier/Classifier.i" 
%include "classifier/LinearClassifier.h" 
