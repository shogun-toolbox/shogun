%{
    #include "clustering/KMeans.h"
%}

#ifdef HAVE_PYTHON
%feature("autodoc", "get_radi(self) -> numpy 1dim array of float") get_radi;
%apply (DREAL** ARGOUT1, INT* DIM1) {(DREAL** radi, INT* num)};
%feature("autodoc", "get_centers(self) -> numpy 2dim array of float") get_centers;
%apply (DREAL** ARGOUT2, INT* DIM1, INT* DIM2) {(DREAL** centers, INT* dim, INT* num)};
#endif

%rename(KMeans) CKMeans;

%include "clustering/KMeans.h"

