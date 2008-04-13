%{
 #include "lib/io.h"
 #include "features/Labels.h"
%}

#ifdef HAVE_PYTHON
%include "lib/python_typemaps.i"
%feature("autodoc", "get_labels(self) -> numpy 1dim array of float") get_labels;
#endif

#ifdef HAVE_OCTAVE
%include "lib/octave_typemaps.i"
#endif

%apply (DREAL* IN_ARRAY1, INT DIM1) {(DREAL* labels, INT len)};
%apply (DREAL** ARGOUT1, INT* DIM1) {(DREAL** labels, INT* len)};

%rename(Labels) CLabels;

%include "features/Labels.h"

%ignore CLabels::CLabels(INT num_labels);
%ignore CLabels::CLabels(CHAR* fname);
