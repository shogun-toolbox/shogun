%define DOCSTR
"The `Library` module gathers all miscellaneous Objects in the SHOGUN toolkit."
%enddef

%module(docstring=DOCSTR,directors="1") Library
%feature("autodoc","1");

#ifdef USE_DOXYGEN
%include "lib/Library_doxygen.i"
#endif

%include "lib/common.i"
%include "lib/ShogunException.i"
%include "lib/io.i"
%include "base/SGObject.i"

%include "lib/Cache.i"
%include "lib/File.i"
%include "lib/List.i"
%include "lib/Mathematics.i"
%include "lib/Signal.i"
%include "lib/SimpleFile.i"
%include "lib/Time.i"
%include "lib/Trie.i"
%include "lib/DynamicArray.i"
%include "lib/Array.i"
%include "lib/Array2.i"
%include "lib/Array3.i"
