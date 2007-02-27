%{
    #include "features/SimpleFeatures.h" 
%}

#ifdef HAVE_PYTHON
%include "lib/numpy.i"
%apply (ST* IN_ARRAY2, INT DIM1, INT DIM2) {(ST* src, INT num_feat, INT num_vec)};
#endif

%include "features/SimpleFeatures.h" 

%template(SimpleRealFeatures) CSimpleFeatures<DREAL>;
%template(SimpleByteFeatures) CSimpleFeatures<BYTE>;
%template(SimpleWordFeatures) CSimpleFeatures<WORD>;
%template(SimpleShortFeatures) CSimpleFeatures<SHORT>;
%template(SimpleCharFeatures) CSimpleFeatures<CHAR>;
%template(SimpleIntFeatures)  CSimpleFeatures<INT>;
