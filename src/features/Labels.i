%{
 #include "lib/io.h"
 #include "features/Labels.h"
%}

%include "lib/common.i"
%include "lib/swig_typemaps.i"

#ifdef HAVE_PYTHON
%feature("autodoc", "get_labels(self) -> numpy 1dim array of float") get_labels;
#endif

%apply (DREAL* IN_ARRAY1, INT DIM1) {(DREAL* labels, INT len)};
%apply (DREAL** ARGOUT1, INT* DIM1) {(DREAL** labels, INT* len)};

%ignore CLabels::CLabels();
%ignore CLabels::CLabels(INT num_labels);
%ignore CLabels::CLabels(CHAR* fname);

%rename(Labels) CLabels;

%include "features/Labels.h"
