%{
 #include "features/SparseFeatures.h" 
%}

%include "features/SparseFeatures.h" 

%template(SparseCharFeatures) CSparseFeatures<char>;
%template(SparseByteFeatures) CSparseFeatures<uint8_t>;
%template(SparseShortFeatures) CSparseFeatures<int16_t>;
%template(SparseWordFeatures) CSparseFeatures<uint16_t>;
%template(SparseIntFeatures) CSparseFeatures<int32_t>;
%template(SparseUIntFeatures) CSparseFeatures<uint32_t>;
%template(SparseLongFeatures) CSparseFeatures<int64_t>;
%template(SparseUlongFeatures) CSparseFeatures<uint64_t>;
%template(SparseRealFeatures) CSparseFeatures<DREAL>;
%template(SparseShortRealFeatures) CSparseFeatures<SHORTREAL>;
%template(SparseLongRealFeatures) CSparseFeatures<LONGREAL>;
