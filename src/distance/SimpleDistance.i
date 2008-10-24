%{
 #include "distance/SimpleDistance.h"
%}

%include "lib/common.i"
%include "distance/Distance.h"
%include "distance/SimpleDistance.h"

%template(SimpleRealDistance) CSimpleDistance<DREAL>;
%template(SimpleWordDistance) CSimpleDistance<WORD>;
%template(SimpleCharDistance) CSimpleDistance<char>;
%template(SimpleIntDistance) CSimpleDistance<INT>;

