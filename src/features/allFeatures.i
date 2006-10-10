%module(directors="1") allFeatures

%feature("director");

%{
    #include "features/Features.h" 
    #include "features/SimpleFeatures.h" 
    #include "features/RealFeatures.h" 
%}

