%{
    #include <shogun/clustering/KMeans.h>
%}

%include "lib/swig_typemaps.i"

#ifdef HAVE_PYTHON
%feature("autodoc", "get_radi(self) -> numpy 1dim array of float") get_radi;
%feature("autodoc", "get_centers(self) -> numpy 2dim array of float") get_centers;
#endif

%apply (float64_t** ARGOUT1, int32_t* DIM1) {(float64_t** radii, int32_t* num)};
%apply (float64_t** ARGOUT2, int32_t* DIM1, int32_t* DIM2) {(float64_t** centers, int32_t* dim, int32_t* num)};

%rename(KMeans) CKMeans;

%include <shogun/clustering/KMeans.h>

