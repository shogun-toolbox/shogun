%module RealFeatures

%{
    #include "features/RealFeatures.h" 
%}

%include "lib/common.i"
%include "features/SimpleFeatures.i"

%feature("notabstract") CRealFeatures;
%include "features/RealFeatures.h"
