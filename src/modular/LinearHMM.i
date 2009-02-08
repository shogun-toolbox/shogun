%{
#include <shogun/distributions/hmm/LinearHMM.h>
%}

#ifdef HAVE_PYTHON
%feature("autodoc", "get_log_transition_probs(self) -> numpy 1dim array of %float") get_log_transition_probs;
%feature("autodoc", "get_transition_probs(self) -> numpy 1dim array of %float") get_transition_probs;
%apply (float64_t** ARGOUT1, int32_t* DIM1) {(float64_t** dst, int32_t* num)};
%apply (float64_t* IN_ARRAY1, int32_t DIM1) {(const float64_t* src, int32_t num)};
#endif

%rename(LinearHMM) CLinearHMM;

%include <shogun/distributions/hmm/LinearHMM.h>
