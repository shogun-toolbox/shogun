%{
    #include "features/SimpleFeatures.h" 
%}

#ifdef HAVE_PYTHON
%include "lib/python_typemaps.i"
#endif

%include "features/SimpleFeatures.h" 

%template(SimpleRealFeatures) CSimpleFeatures<DREAL>;
%template(SimpleByteFeatures) CSimpleFeatures<BYTE>;
%template(SimpleWordFeatures) CSimpleFeatures<WORD>;
%template(SimpleShortFeatures) CSimpleFeatures<SHORT>;
%template(SimpleCharFeatures) CSimpleFeatures<CHAR>;
%template(SimpleIntFeatures)  CSimpleFeatures<INT>;
