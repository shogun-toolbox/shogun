%{
    #include "clustering/Hierarchical.h"
%}

#ifdef HAVE_PYTHON
%feature("autodoc", "get_merge_distance(self) -> [] of float") get_merge_distance;
%apply (DREAL** ARGOUT1, INT* DIM1) {(DREAL** dist, INT* num)};
%feature("autodoc", "get_pairs(self) -> [] of float") get_pairs;
%apply (INT** ARGOUT2, INT* DIM1, INT* DIM2) {(INT** tuples, INT* rows, INT* num)};
#endif

%rename(Hierarchical) CHierarchical;

%include "clustering/Hierarchical.h"

