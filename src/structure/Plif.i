%{
 #include "structure/Plif.h" 
%}

%rename(Plif) CPlif;

#ifdef HAVE_PYTHON
%apply (DREAL* IN_ARRAY1, int32_t DIM1) {(DREAL* p_limits, int32_t p_len)};
%apply (DREAL* IN_ARRAY1, int32_t DIM1) {(DREAL* p_penalties, int32_t p_len)};
#endif

%include "structure/PlifBase.i" 
%include "structure/Plif.h" 
