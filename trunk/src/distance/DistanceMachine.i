 %module(directors="1") CDistanceMachine
%{
 #include "distance/DistanceMachine.h"
%}

%include "lib/common.i"

%feature("director") CDistanceMachine;

%include "distance/DistanceMachine.h"
%include "distance/DistanceMachine.h"

