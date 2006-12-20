%module Structure
%{
#define SWIG_FILE_WITH_INIT
%}

%include "lib/common.i"

#ifdef HAVE_PYTHON
%init %{
	  import_array();
%}

%include "lib/numpy.i"
#endif

%include "structure/PlifBase.i"
%include "structure/Plif.i"
%include "structure/PlifArray.i"
%include "structure/DynProg.i"
