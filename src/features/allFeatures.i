%module(directors="1") allFeatures

%feature("director");

%{
    #include "features/Features.h" 
    #include "features/SimpleFeatures.h" 
    #include "features/RealFeatures.h" 
%}

%include "features/Features.i" 
%include "features/SimpleFeatures.i" 
%include "features/RealFeatures.i" 

