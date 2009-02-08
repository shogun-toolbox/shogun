%{
    #include <shogun/distance/SimpleDistance.h>
    #include <shogun/distance/RealDistance.h>
%}

%rename(RealDistance) CRealDistance;

%include "SimpleDistance.i"
%include <shogun/distance/RealDistance.h>
