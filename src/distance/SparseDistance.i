%{
 #include "distance/SparseDistance.h"
%}

%include "lib/common.i"
%include "distance/Distance.h"
%include "distance/SparseDistance.h"

%template(SparseRealDistance) CSparseDistance<DREAL>;
%template(SparseWordDistance) CSparseDistance<WORD>;
%template(SparseCharDistance) CSparseDistance<char>;
%template(SparseIntDistance) CSparseDistance<INT>;
