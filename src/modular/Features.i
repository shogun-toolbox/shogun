%define DOCSTR
"The `Features` module gathers all Feature objects available in the SHOGUN toolkit."
%enddef

%module(docstring=DOCSTR) Features

%{
#define SWIG_FILE_WITH_INIT
#include <shogun/features/Features.h>
#ifdef HAVE_R
#include <Rdefines.h>
#endif
%}

#ifdef HAVE_DOXYGEN
%include "Features_doxygen.i"
#endif

#ifdef HAVE_PYTHON
%init %{
	  import_array();
%}
#endif

%rename(Features) CFeatures;
%feature("autodoc","0");

%include "common.i"
%include "ShogunException.i"
%include "io.i" 
%include "Version.i"
%include "Parallel.i"
%include "SGObject.i"
%include <shogun/features/Features.h>
%include "StringFeatures.i"
%include "DotFeatures.i" 
%include "SparseFeatures.i"
%include "SimpleFeatures.i" 
%include "DummyFeatures.i" 

%include "Alphabet.i"
%include "CharFeatures.i"
%include "ByteFeatures.i"
%include "ShortFeatures.i"
%include "WordFeatures.i"
%include "RealFeatures.i"
%include "ShortRealFeatures.i"
%include "CombinedFeatures.i"
%include "CombinedDotFeatures.i"
%include "MindyGramFeatures.i"
%include "Labels.i"

%include "RealFileFeatures.i"
%include "FKFeatures.i"
%include "TOPFeatures.i"
%include "WDFeatures.i"
%include "ExplicitSpecFeatures.i"
%include "ImplicitWeightedSpecFeatures.i"
