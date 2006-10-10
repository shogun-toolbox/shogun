%{
 #include "features/StringFeatures.h" 
%}

%include "features/StringFeatures.h" 

%template(StringRealFeatures) CStringFeatures<DREAL>;
%template(StringCharFeatures) CStringFeatures<CHAR>;
%template(StringUlongFeatures) CStringFeatures<ULONG>;
%template(StringWordFeatures) CStringFeatures<WORD>;
