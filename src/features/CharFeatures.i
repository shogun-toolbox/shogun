%{
#include "features/CharFeatures.h" 
%}

%include "lib/common.i"

%apply (char* IN_ARRAY2, int DIM1, int DIM2) {(char *feature_matrix, int d1, int d2)};
%apply (int* IN_ARRAY2, int DIM1, int DIM2) {(int *feature_matrix, int d1, int d2)};

%apply (double* IN_ARRAY1, int DIM1) {(double* series, int size)};

%apply (int PARA, char* IN_ARRAY2, int DIM1, int DIM2) {(int alphabet, char* feature_matrix, int num_feat, INT num_vec)};

%apply (int PARA, double* IN_ARRAY2, int DIM1, int DIM2) {(int alphabet, char* feature_matrix, int num_feat, INT num_vec)};

%apply (double* IN_ARRAY2, int DIM1, int DIM2, int PARA ) {(char* feature_matrix, int num_feat, INT num_vec, int alphabet)};
%include "features/CharFeatures.h"
