%{
#include "distributions/hmm/LinearHMM.h"
%}

#ifdef HAVE_PYTHON
%feature("autodoc", "get_log_transition_probs(self) -> numpy 1dim array of %float") get_log_transition_probs;
%feature("autodoc", "get_transition_probs(self) -> numpy 1dim array of %float") get_transition_probs;
%apply (DREAL** ARGOUT1, int32_t* DIM1) {(DREAL** dst, int32_t* num)};
%apply (DREAL* IN_ARRAY1, int32_t DIM1) {(const DREAL* src, int32_t num)};
#endif

%rename(LinearHMM) CLinearHMM;

%include "distributions/hmm/LinearHMM.h"

