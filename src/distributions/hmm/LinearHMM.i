%{
#include "distributions/hmm/LinearHMM.h"
%}

#ifdef HAVE_PYTHON
%apply (DREAL** ARGOUT1, INT* DIM1) {(DREAL** dst, INT* num)};
%apply (DREAL* IN_ARRAY1, INT DIM1) {(const DREAL* src, INT num)};
#endif

%rename(LinearHMM) CLinearHMM;

%include "distributions/hmm/LinearHMM.h"

