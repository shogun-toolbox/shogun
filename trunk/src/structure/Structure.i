%module Structure
%{
#define SWIG_FILE_WITH_INIT
%}

%include "lib/common.i"

%init %{
	  import_array();
%}

%include "lib/numpy.i"

%include "structure/DynProg.i"
%include "structure/Plif.i"
