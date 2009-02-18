%module Structure
%{
#define SWIG_FILE_WITH_INIT
%}

#ifdef HAVE_DOXYGEN
%include "Structure_doxygen.i"
#endif

#ifdef HAVE_PYTHON
%init %{
	  import_array();
%}

%include "swig_typemaps.i"
#endif

%include "init.i"
%include "common.i"
%include "ShogunException.i"
%include "io.i" 
%include "Version.i"
%include "Parallel.i"
%include "SGObject.i"

%include "PlifBase.i"
%include "Plif.i"
%include "PlifArray.i"
%include "DynProg.i"
