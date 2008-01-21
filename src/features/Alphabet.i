%{
#include "features/Alphabet.h"
%}

#ifdef HAVE_PYTHON
%include "lib/python_typemaps.i"
%feature("autodoc", "get_hist(self) -> numpy 1dim array of int") get_hist;
%apply (LONG** ARGOUT1, INT* DIM1) {(LONG** h, INT* len)};
#endif

%rename(Alphabet) CAlphabet;

%include "features/Alphabet.h"
