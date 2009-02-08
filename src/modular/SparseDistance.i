%{
 #include <shogun/distance/SparseDistance.h>
%}

%include "common.i"
%include <shogun/distance/Distance.h>
%include <shogun/distance/SparseDistance.h>

%template(SparseRealDistance) CSparseDistance<float64_t>;
%template(SparseWordDistance) CSparseDistance<uint16_t>;
%template(SparseCharDistance) CSparseDistance<char>;
%template(SparseIntDistance) CSparseDistance<int32_t>;
