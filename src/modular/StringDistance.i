%{
 #include <shogun/distance/StringDistance.h>
%}

%include <shogun/distance/StringDistance.h>

%template(StringRealDistance) CStringDistance<float64_t>;
%template(StringWordDistance) CStringDistance<uint16_t>;
%template(StringCharDistance) CStringDistance<char>;
%template(StringIntDistance) CStringDistance<int32_t>;
%template(StringUlongDistance) CStringDistance<uint64_t>;
