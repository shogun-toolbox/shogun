%define DOCSTR
"The `Features` module gathers all Feature objects available in the SHOGUN toolkit."
%enddef

%module(docstring=DOCSTR,directors="1") Features

%{
#define SWIG_FILE_WITH_INIT
#include "features/Features.h" 
#include "features/StringFeatures.h" 
%}

#ifdef HAVE_PYTHON
%init %{
	  import_array();
%}
#endif

%feature("director") CFeatures;
%rename(Features) CFeatures;
%feature("autodoc","1");

%include "lib/common.i"
%include "lib/io.i" 
%include "base/Version.i"
%include "base/Parallel.i"
%include "base/SGObject.i"
%include "features/Features.h" 

%include "features/Alphabet.i"
%include "features/CharFeatures.i"
%include "features/ByteFeatures.i"
%include "features/ShortFeatures.i"
%include "features/WordFeatures.i"
%include "features/RealFeatures.i"
%include "features/StringFeatures.i"
%include "features/Labels.i"
