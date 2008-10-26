%{
 #include "distance/StringDistance.h" 
%}

%include "distance/StringDistance.h" 

%template(StringRealDistance) CStringDistance<DREAL>;
%template(StringWordDistance) CStringDistance<uint16_t>;
%template(StringCharDistance) CStringDistance<char>;
%template(StringIntDistance) CStringDistance<int32_t>;
%template(StringUlongDistance) CStringDistance<uint64_t>;
