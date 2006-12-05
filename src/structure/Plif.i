%{
 #include "structure/Plif.h" 
%}

%rename(Plif) CPlif;

%apply (DREAL* IN_ARRAY1, INT DIM1) {(DREAL* p_limits, INT p_len)};
%apply (DREAL* IN_ARRAY1, INT DIM1) {(DREAL* p_penalties, INT p_len)};

%include "structure/Plif.h" 
