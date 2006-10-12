%{
    #include "features/SimpleFeatures.h" 
%}

%include "lib/numpy.i"

%apply (CHAR* IN_ARRAY2, INT DIM1, INT DIM2) {(CHAR* fm, INT num_feat, INT num_vec)};
%apply (BYTE* IN_ARRAY2, INT DIM1, INT DIM2) {(BYTE* fm, INT num_feat, INT num_vec)};
%apply (WORD* IN_ARRAY2, INT DIM1, INT DIM2) {(WORD* fm, INT num_feat, INT num_vec)};
%apply (SHORT* IN_ARRAY2, INT DIM1, INT DIM2) {(SHORT* fm, INT num_feat, INT num_vec)};
%apply (INT* IN_ARRAY2, INT DIM1, INT DIM2) {(INT* fm, INT num_feat, INT num_vec)};
%apply (DREAL* IN_ARRAY2, INT DIM1, INT DIM2) {(DREAL* fm, INT num_feat, INT num_vec)};
%apply (ULONG* IN_ARRAY2, INT DIM1, INT DIM2) {(ULONG* fm, INT num_feat, INT num_vec)};

%include "features/SimpleFeatures.h" 

%template(SimpleRealFeatures) CSimpleFeatures<DREAL>;
%template(SimpleByteFeatures) CSimpleFeatures<BYTE>;
%template(SimpleWordFeatures) CSimpleFeatures<WORD>;
%template(SimpleShortFeatures) CSimpleFeatures<SHORT>;
%template(SimpleCharFeatures) CSimpleFeatures<CHAR>;
%template(SimpleIntFeatures)  CSimpleFeatures<INT>;
