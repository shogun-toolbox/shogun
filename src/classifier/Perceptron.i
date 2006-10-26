%{
 #include "classifier/Perceptron.h" 
%}

%rename(Perceptron) CPerceptron;

%include "classifier/LinearClassifier.i" 
%include "classifier/Perceptron.h" 
