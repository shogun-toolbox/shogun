%{
 #include "distance/SimpleDistance.h"
%}

%include "lib/common.i"
%include "distance/Distance.h"
%include "distance/SimpleDistance.h"

%template(SimpleRealDistance) CSimpleDistance<DREAL>;
%template(SimpleWordDistance) CSimpleDistance<uint16_t>;
%template(SimpleCharDistance) CSimpleDistance<char>;
%template(SimpleIntDistance) CSimpleDistance<int32_t>;

