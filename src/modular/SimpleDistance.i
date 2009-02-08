%{
 #include <shogun/distance/SimpleDistance.h>
%}

%include "common.i"
%include <shogun/distance/Distance.h>
%include <shogun/distance/SimpleDistance.h>

%template(SimpleRealDistance) CSimpleDistance<float64_t>;
%template(SimpleWordDistance) CSimpleDistance<uint16_t>;
%template(SimpleCharDistance) CSimpleDistance<char>;
%template(SimpleIntDistance) CSimpleDistance<int32_t>;

