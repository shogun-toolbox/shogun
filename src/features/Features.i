%define DOCSTR
"The `Features` module gathers all Feature objects available in the SHOGUN toolkit."
%enddef

%module(docstring=DOCSTR) Features

%{
#define SWIG_FILE_WITH_INIT
#include "features/Features.h" 
#ifdef HAVE_R
#include <Rdefines.h>
#endif
%}

#ifdef HAVE_DOXYGEN
%include "features/Features_doxygen.i"
#endif

#ifdef HAVE_PYTHON
%init %{
	  import_array();
%}
#endif

%rename(Features) CFeatures;
%feature("autodoc","0");

%include "lib/common.i"
%include "lib/ShogunException.i"
%include "lib/io.i" 
%include "base/Version.i"
%include "base/Parallel.i"
%include "base/SGObject.i"
%include "features/Features.h" 
%include "features/StringFeatures.i"
%include "features/SparseFeatures.i"
%include "features/SimpleFeatures.i" 

%include "features/Alphabet.i"
%include "features/CharFeatures.i"
%include "features/ByteFeatures.i"
%include "features/ShortFeatures.i"
%include "features/WordFeatures.i"
%include "features/RealFeatures.i"
%include "features/ShortRealFeatures.i"
%include "features/CombinedFeatures.i"
%include "features/MindyGramFeatures.i"
%include "features/Labels.i"

%include "features/RealFileFeatures.i"
%include "features/FKFeatures.i"
%include "features/TOPFeatures.i"
