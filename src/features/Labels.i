%{
 #include "lib/io.h"
 #include "features/Labels.h"
%}

#ifdef HAVE_PYTHON
%include "lib/python_typemaps.i"
%apply (DREAL* IN_ARRAY1, INT DIM1) {(DREAL* labels, INT len)};
%feature("autodoc", "get_labels(self) -> numpy 1dim array of float") get_labels;
%apply (DREAL** ARGOUT1, INT* DIM1) {(DREAL** labels, INT* len)};
#endif

%include "features/Labels.h"

%ignore CLabels::CLabels(INT num_labels);
%ignore CLabels::CLabels(CHAR* fname);

#ifdef HAVE_PYTHON
%pythoncode %{
  class Labels(CLabels):
     def __init__(self,p1):
        CLabels.__init__(self)
        self.set_labels(p1)
%}
#endif
