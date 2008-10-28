%{
 #include "distance/SparseDistance.h"
%}

%include "lib/common.i"
%include "distance/Distance.h"
%include "distance/SparseDistance.h"

%template(SparseRealDistance) CSparseDistance<float64_t>;
%template(SparseWordDistance) CSparseDistance<uint16_t>;
%template(SparseCharDistance) CSparseDistance<char>;
%template(SparseIntDistance) CSparseDistance<int32_t>;
