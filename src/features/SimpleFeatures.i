%{
    #include "features/SimpleFeatures.h" 
%}

%include "lib/numpy.i"

%apply (CHAR* IN_ARRAY2, INT DIM1, INT DIM2) {(CHAR* src, INT num_feat, INT num_vec)};
%apply (BYTE* IN_ARRAY2, INT DIM1, INT DIM2) {(BYTE* src, INT num_feat, INT num_vec)};
%apply (WORD* IN_ARRAY2, INT DIM1, INT DIM2) {(WORD* src, INT num_feat, INT num_vec)};
%apply (SHORT* IN_ARRAY2, INT DIM1, INT DIM2) {(SHORT* src, INT num_feat, INT num_vec)};
%apply (INT* IN_ARRAY2, INT DIM1, INT DIM2) {(INT* src, INT num_feat, INT num_vec)};
%apply (DREAL* IN_ARRAY2, INT DIM1, INT DIM2) {(DREAL* src, INT num_feat, INT num_vec)};
%apply (ULONG* IN_ARRAY2, INT DIM1, INT DIM2) {(ULONG* src, INT num_feat, INT num_vec)};

%include "features/SimpleFeatures.h" 

%template(SimpleRealFeatures) CSimpleFeatures<DREAL>;
%template(SimpleByteFeatures) CSimpleFeatures<BYTE>;
%template(SimpleWordFeatures) CSimpleFeatures<WORD>;
%template(SimpleShortFeatures) CSimpleFeatures<SHORT>;
%template(SimpleCharFeatures) CSimpleFeatures<CHAR>;
%template(SimpleIntFeatures)  CSimpleFeatures<INT>;
