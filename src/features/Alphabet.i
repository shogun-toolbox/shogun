%{
#include "features/Alphabet.h"
%}

%include "lib/swig_typemaps.i"

#ifdef HAVE_PYTHON
%feature("autodoc", "get_hist(self) -> numpy 1dim array of int") get_hist;
#endif

%apply (LONG** ARGOUT1, INT* DIM1) {(LONG** h, INT* len)};

%rename(Alphabet) CAlphabet;

%include "features/Alphabet.h"
