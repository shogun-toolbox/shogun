%{
    #include "distance/SimpleDistance.h" 
    #include "distance/RealDistance.h" 
%}

%rename(RealDistance) CRealDistance;

%include "distance/SimpleDistance.i"
%include "distance/RealDistance.h"
