%{
 #include "distance/StringDistance.h" 
%}

%include "distance/StringDistance.h" 

%template(StringRealDistance) CStringDistance<DREAL>;
%template(StringWordDistance) CStringDistance<uint16_t>;
%template(StringCharDistance) CStringDistance<char>;
%template(StringIntDistance) CStringDistance<INT>;
%template(StringUlongDistance) CStringDistance<ULONG>;
