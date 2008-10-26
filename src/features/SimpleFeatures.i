%{
    #include "features/SimpleFeatures.h" 
%}

%include "features/SimpleFeatures.h" 

%template(SimpleRealFeatures) CSimpleFeatures<DREAL>;
%template(SimpleShortRealFeatures) CSimpleFeatures<SHORTREAL>;
%template(SimpleByteFeatures) CSimpleFeatures<uint8_t>;
%template(SimpleWordFeatures) CSimpleFeatures<uint16_t>;
%template(SimpleShortFeatures) CSimpleFeatures<SHORT>;
%template(SimpleCharFeatures) CSimpleFeatures<char>;
%template(SimpleIntFeatures)  CSimpleFeatures<int32_t>;

