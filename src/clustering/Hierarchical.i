%{
    #include "clustering/Hierarchical.h"
%}

#ifdef HAVE_PYTHON
%apply (DREAL** ARGOUT1, INT* DIM1) {(DREAL** dist, INT* num)};
%apply (INT** ARGOUT2, INT* DIM1, INT* DIM2) {(INT** tuples, INT* rows, INT* num)};
#endif

%rename(Hierarchical) CHierarchical;

%include "clustering/Hierarchical.h"

