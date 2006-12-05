%{
 #include "classifier/LDA.h" 
%}

%rename(LDA) CLDA;

%include "classifier/LinearClassifier.i" 
%include "classifier/LDA.h" 
