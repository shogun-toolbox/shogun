%{
 #include <shogun/structure/Plif.h>
%}

%rename(Plif) CPlif;

#ifdef HAVE_PYTHON
%apply (float64_t* IN_ARRAY1, int32_t DIM1) {(float64_t* p_limits, int32_t p_len)};
%apply (float64_t* IN_ARRAY1, int32_t DIM1) {(float64_t* p_penalties, int32_t p_len)};
#endif

%include "PlifBase.i" 
%include <shogun/structure/Plif.h>
