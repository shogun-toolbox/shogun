%{
    #include "features/SimpleFeatures.h" 
%}

%include "lib/swig_typemaps.i"

%include "features/SimpleFeatures.h" 

%template(SimpleRealFeatures) CSimpleFeatures<DREAL>;
%template(SimpleByteFeatures) CSimpleFeatures<BYTE>;
%template(SimpleWordFeatures) CSimpleFeatures<WORD>;
%template(SimpleShortFeatures) CSimpleFeatures<SHORT>;
%template(SimpleCharFeatures) CSimpleFeatures<CHAR>;
%template(SimpleIntFeatures)  CSimpleFeatures<INT>;
