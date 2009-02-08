%{
#include <shogun/features/Alphabet.h>
%}

%include "swig_typemaps.i"

#ifdef HAVE_PYTHON
%feature("autodoc", "get_hist(self) -> numpy 1dim array of int") get_hist;
#endif

%apply (int64_t** ARGOUT1, int32_t* DIM1) {(int64_t** h, int32_t* len)};

%rename(Alphabet) CAlphabet;

%include <shogun/features/Alphabet.h>
