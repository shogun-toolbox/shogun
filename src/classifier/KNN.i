%{
 #include "distance/DistanceMachine.h"
 #include "classifier/KNN.h" 
%}

%rename(KNN) CKNN;

%include "distance/DistanceMachine.i"
%include "classifier/KNN.h" 
