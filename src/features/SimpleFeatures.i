%{
    #include "features/SimpleFeatures.h" 
%}

%include "features/SimpleFeatures.h" 

%template(SimpleRealFeatures) CSimpleFeatures<DREAL>;
%template(SimpleShortRealFeatures) CSimpleFeatures<SHORTREAL>;
%template(SimpleByteFeatures) CSimpleFeatures<BYTE>;
%template(SimpleWordFeatures) CSimpleFeatures<WORD>;
%template(SimpleShortFeatures) CSimpleFeatures<SHORT>;
%template(SimpleCharFeatures) CSimpleFeatures<char>;
%template(SimpleIntFeatures)  CSimpleFeatures<INT>;

