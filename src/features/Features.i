%module(directors="1") Features

%{
#define SWIG_FILE_WITH_INIT
#include "features/Features.h" 
#include "features/StringFeatures.h" 
%}

%include "lib/common.i"
%include "lib/numpy.i"

%init %{
	  import_array();
%}

%feature("director") CFeatures;

%apply (CHAR* IN_ARRAY1, INT DIM1) {(CHAR*, INT)};
%apply (BYTE* IN_ARRAY1, INT DIM1) {(BYTE*, INT)};
%apply (WORD* IN_ARRAY1, INT DIM1) {(WORD*, INT)};
%apply (INT* IN_ARRAY1, INT DIM1) {(INT*, INT)};
%apply (DREAL* IN_ARRAY1, INT DIM1) {(DREAL*, INT)}; 
%apply (DREAL* IN_ARRAY1, INT DIM1) {(DREAL* labels, INT len)};
%apply (double*, int) {(double*, int)};
%apply (ULONG* IN_ARRAY1, INT DIM1) {(ULONG*, INT)};

%apply (CHAR* IN_ARRAY2, INT DIM1, INT DIM2) {(CHAR*, INT, INT)};
%apply (BYTE* IN_ARRAY2, INT DIM1, INT DIM2) {(BYTE*, INT, INT)};
%apply (WORD* IN_ARRAY2, INT DIM1, INT DIM2) {(WORD*, INT, INT)};
%apply (INT* IN_ARRAY2, INT DIM1, INT DIM2) {(INT*, INT, INT)};
%apply (DREAL* IN_ARRAY2, INT DIM1, INT DIM2) {(DREAL*, INT, INT)};
%apply (ULONG* IN_ARRAY2, INT DIM1, INT DIM2) {(ULONG*, INT, INT)};

%include "features/Features.h" 

%include "features/Alphabet.i"
%include "features/ByteFeatures.i"
%include "features/RealFeatures.i"
%include "features/Labels.i"
%include "features/CharFeatures.i"

/*
%include "features/CombinedFeatures.i"
%include "features/Features.i"
%include "features/FKFeatures.i"
%include "features/MindyGramFeatures.i"
%include "features/RealFileFeatures.i"
%include "features/ShortFeatures.i"
%include "features/SimpleFeatures.i"
%include "features/SparseFeatures.i"
%include "features/SparseRealFeatures.i"
%include "features/StringFeatures.i"
%include "features/TOPFeatures.i"
%include "features/WordFeatures.i"
*/

