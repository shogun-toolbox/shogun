%module Features

%{
#include "features/Features.h" 
#include "features/StringFeatures.h" 
%}


/*
%include "features/SimpleFeatures.i" 
%include "features/StringFeatures.i" 
%include "features/Features.i" 
%include "features/SimpleFeatures.i" 
%include "features/RealFeatures.i" 
%include "features/CharFeatures.i" 
%include "features/StringFeatures.i" 
*/

%include "features/allFeatures.i"
%include "features/Alphabet.i"
%include "features/ByteFeatures.i"
%include "features/CharFeatures.i"
%include "features/CombinedFeatures.i"
%include "features/Features.i"
%include "features/FKFeatures.i"
%include "features/Labels.i"
%include "features/MindyGramFeatures.i"
%include "features/RealFeatures.i"
%include "features/RealFileFeatures.i"
%include "features/ShortFeatures.i"
%include "features/SimpleFeatures.i"
%include "features/SparseFeatures.i"
%include "features/SparseRealFeatures.i"
%include "features/StringFeatures.i"
%include "features/TOPFeatures.i"
%include "features/WordFeatures.i"
