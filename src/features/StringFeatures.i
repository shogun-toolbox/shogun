%{
 #include "features/StringFeatures.h" 
%}

%include "features/StringFeatures.h" 

%template(StringCharFeatures) CStringFeatures<CHAR>;
%template(StringByteFeatures) CStringFeatures<BYTE>;
%template(StringWordFeatures) CStringFeatures<WORD>;
%template(StringUlongFeatures) CStringFeatures<ULONG>;
