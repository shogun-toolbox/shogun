%module CharFeatures

%{
    #include "features/CharFeatures.h" 
%}

%include "lib/common.i"
%include "features/Alphabet.i"
%include "features/SimpleFeatures.i"

%feature("notabstract") CCharFeatures;
