%{
    #include "clustering/KMeans.h" 
%}

#ifdef HAVE_PYTHON
%apply (DREAL** ARGOUT1, INT* DIM1) {(DREAL** radi, INT* num)};
%apply (DREAL** ARGOUT2, INT* DIM1, INT* DIM2) {(DREAL** centers, INT* dim, INT* num)};
#endif

%rename(KMeans) CKMeans;

%include "clustering/KMeans.h"

