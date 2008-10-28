%{
 #include "lib/io.h"
 #include "features/Labels.h"
%}

%include "lib/common.i"
%include "lib/swig_typemaps.i"

#ifdef HAVE_PYTHON
%feature("autodoc", "get_labels(self) -> numpy 1dim array of float") get_labels;
#endif

%apply (float64_t* IN_ARRAY1, int32_t DIM1) {(float64_t* labels, int32_t len)};
%apply (float64_t** ARGOUT1, int32_t* DIM1) {(float64_t** labels, int32_t* len)};

%rename(Labels) CLabels;

%include "features/Labels.h"
