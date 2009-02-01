%{
    #include <shogun/clustering/Hierarchical.h>
%}

#ifdef HAVE_PYTHON
%feature("autodoc", "get_merge_distance(self) -> [] of float") get_merge_distance;
%feature("autodoc", "get_pairs(self) -> [] of float") get_pairs;
#endif

%apply (float64_t** ARGOUT1, int32_t* DIM1) {(float64_t** dist, int32_t* num)};
%apply (int32_t** ARGOUT2, int32_t* DIM1, int32_t* DIM2) {(int32_t** tuples, int32_t* rows, int32_t* num)};

%rename(Hierarchical) CHierarchical;

%include <shogun/clustering/Hierarchical.h>

