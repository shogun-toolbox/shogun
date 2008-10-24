%{
 #include "features/SparseFeatures.h" 
%}

%include "features/SparseFeatures.h" 

%template(SparseCharFeatures) CSparseFeatures<char>;
%template(SparseByteFeatures) CSparseFeatures<uint8_t>;
%template(SparseShortFeatures) CSparseFeatures<SHORT>;
%template(SparseWordFeatures) CSparseFeatures<WORD>;
%template(SparseIntFeatures) CSparseFeatures<INT>;
%template(SparseUIntFeatures) CSparseFeatures<UINT>;
%template(SparseLongFeatures) CSparseFeatures<LONG>;
%template(SparseUlongFeatures) CSparseFeatures<ULONG>;
%template(SparseRealFeatures) CSparseFeatures<DREAL>;
%template(SparseShortRealFeatures) CSparseFeatures<SHORTREAL>;
%template(SparseLongRealFeatures) CSparseFeatures<LONGREAL>;
