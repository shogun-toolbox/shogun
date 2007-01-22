%{
 #include "distance/StringDistance.h" 
%}

%include "distance/StringDistance.h" 

%template(StringRealDistance) CStringDistance<DREAL>;
%template(StringWordDistance) CStringDistance<WORD>;
%template(StringCharDistance) CStringDistance<CHAR>;
%template(StringIntDistance) CStringDistance<INT>;
%template(StringUlongDistance) CStringDistance<ULONG>;
