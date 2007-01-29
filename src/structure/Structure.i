%module Structure
%{
#define SWIG_FILE_WITH_INIT
%}


#ifdef HAVE_PYTHON
%init %{
	  import_array();
%}

%include "lib/numpy.i"
#endif

%include "lib/common.i"
%include "lib/ShogunException.i"
%include "lib/io.i" 
%include "base/Version.i"
%include "base/Parallel.i"
%include "base/SGObject.i"

%include "structure/PlifBase.i"
%include "structure/Plif.i"
%include "structure/PlifArray.i"
%include "structure/DynProg.i"
