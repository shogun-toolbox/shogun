%{
    #include "clustering/Hierarchical.h"
%}

#ifdef HAVE_PYTHON
%feature("autodoc", "get_merge_distance(self) -> [] of float") get_merge_distance;
%feature("autodoc", "get_pairs(self) -> [] of float") get_pairs;
#endif

%apply (DREAL** ARGOUT1, INT* DIM1) {(DREAL** dist, INT* num)};
%apply (INT** ARGOUT2, INT* DIM1, INT* DIM2) {(INT** tuples, INT* rows, INT* num)};

%rename(Hierarchical) CHierarchical;

%include "clustering/Hierarchical.h"

