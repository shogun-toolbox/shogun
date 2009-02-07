%{
 #include "distance/SimpleDistance.h"
%}

%include "lib/common.i"
%include "distance/Distance.h"
%include "distance/SimpleDistance.h"

%template(SimpleRealDistance) CSimpleDistance<float64_t>;
%template(SimpleWordDistance) CSimpleDistance<uint16_t>;
%template(SimpleCharDistance) CSimpleDistance<char>;
%template(SimpleIntDistance) CSimpleDistance<int32_t>;

