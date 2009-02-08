%define DOCSTR
"The `Library` module gathers all miscellaneous Objects in the SHOGUN toolkit."
%enddef

%module(docstring=DOCSTR) Library
%feature("autodoc","0");

#ifdef HAVE_DOXYGEN
%include "Library_doxygen.i"
#endif

%include "common.i"
%include "ShogunException.i"
%include "io.i"
%include "SGObject.i"

%include "Cache.i"
%include "File.i"
%include "List.i"
%include "Mathematics.i"
%include "Signal.i"
%include "SimpleFile.i"
%include "Time.i"
%include "Trie.i"
%include "DynamicArray.i"
%include "Array.i"
%include "Array2.i"
%include "Array3.i"
