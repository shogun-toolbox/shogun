%{
    #include <shogun/features/DotFeatures.h>
%}

%apply (float64_t** ARGOUT2, int32_t* DIM1, int32_t* DIM2) {(float64_t** matrix, int32_t* d1, int32_t* d2)};

%rename(DotFeatures) CDotFeatures;

%include <shogun/features/DotFeatures.h>
