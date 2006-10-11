%{
#include "features/CharFeatures.h" 
%}

%include "lib/common.i"

%apply (char* IN_ARRAY2, int DIM1, int DIM2) {(char *feature_matrix, int d1, int d2)};
%apply (int* IN_ARRAY2, int DIM1, int DIM2) {(int *feature_matrix, int d1, int d2)};

%apply (double* IN_ARRAY1, int DIM1) {(double* series, int size)};

/*%apply (char *IN_ARRAY1, int DIM1) {char *feature_matrix, int num_feat};*/

/* CCharFeatures(E_ALPHABET alphabet, CHAR* feature_matrix, INT num_feat, INT num_vec);*/

%include "features/SimpleFeatures.i"
%include "features/CharFeatures.h"

