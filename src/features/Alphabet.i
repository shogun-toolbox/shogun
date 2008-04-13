%{
#include "features/Alphabet.h"
%}

#ifdef HAVE_PYTHON
%include "lib/python_typemaps.i"
%feature("autodoc", "get_hist(self) -> numpy 1dim array of int") get_hist;
#endif

#ifdef HAVE_OCTAVE
%include "lib/octave_typemaps.i"
#endif

%apply (LONG** ARGOUT1, INT* DIM1) {(LONG** h, INT* len)};

%rename(Alphabet) CAlphabet;

%include "features/Alphabet.h"
