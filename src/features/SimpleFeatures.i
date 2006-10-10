%{
    #include "features/SimpleFeatures.h" 
%}

%include "features/SimpleFeatures.h" 

%template(SimpleRealFeatures) CSimpleFeatures<DREAL>;
%template(SimpleByteFeatures) CSimpleFeatures<BYTE>;
%template(SimpleWordFeatures) CSimpleFeatures<WORD>;
%template(SimpleCharFeatures) CSimpleFeatures<CHAR>;
%template(SimpleIntFeatures)  CSimpleFeatures<INT>;
