%module(directors="1") Features

%{
#define SWIG_FILE_WITH_INIT
#include "features/Features.h" 
#include "features/StringFeatures.h" 
%}

%include "lib/common.i"

%init %{
	  import_array();
%}

%feature("director") CFeatures;
%rename(Features) CFeatures;

%include "features/Features.h" 

%include "features/Alphabet.i"
%include "features/CharFeatures.i"
%include "features/ByteFeatures.i"
%include "features/ShortFeatures.i"
%include "features/WordFeatures.i"
%include "features/RealFeatures.i"
%include "features/StringFeatures.i"
%include "features/Labels.i"
