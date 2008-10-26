%{
#include "features/Alphabet.h"
%}

%include "lib/swig_typemaps.i"

#ifdef HAVE_PYTHON
%feature("autodoc", "get_hist(self) -> numpy 1dim array of int") get_hist;
#endif

%apply (LONG** ARGOUT1, int32_t* DIM1) {(LONG** h, int32_t* len)};

%rename(Alphabet) CAlphabet;

%include "features/Alphabet.h"
