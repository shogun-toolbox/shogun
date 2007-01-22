%{
 #include "distance/SimpleDistance.h"
%}

%include "lib/common.i"
%include "distance/SimpleDistance.h"

%template(SimpleRealDistance) CSimpleDistance<DREAL>;
%template(SimpleWordDistance) CSimpleDistance<WORD>;
%template(SimpleCharDistance) CSimpleDistance<CHAR>;
%template(SimpleIntDistance) CSimpleDistance<INT>;

