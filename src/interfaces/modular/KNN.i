%{
 #include <shogun/distance/DistanceMachine.h>
 #include <shogun/classifier/KNN.h>
%}

%rename(KNN) CKNN;

%include "distance/DistanceMachine.i"
%include <shogun/classifier/KNN.h>
