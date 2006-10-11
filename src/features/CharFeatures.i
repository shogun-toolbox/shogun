%{
    #include "features/CharFeatures.h" 
%}

%include "lib/numpy.i"
%include "lib/common.i"

%apply (char* IN_ARRAY2, int DIM1, int DIM2) {(char *feature_matrix, int d, int d2)};

/*%apply (char *IN_ARRAY1, int DIM1) {char *feature_matrix, int num_feat};*/

/* CCharFeatures(E_ALPHABET alphabet, CHAR* feature_matrix, INT num_feat, INT num_vec);*/

%include "features/SimpleFeatures.i"
%include "features/CharFeatures.h"

