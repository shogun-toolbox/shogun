%{
 #include "features/SparseFeatures.h" 
%}

%include "features/SparseFeatures.h" 

%template(SparseCharFeatures) CSparseFeatures<CHAR>;
%template(SparseByteFeatures) CSparseFeatures<BYTE>;
%template(SparseShortFeatures) CSparseFeatures<SHORT>;
%template(SparseWordFeatures) CSparseFeatures<WORD>;
%template(SparseIntFeatures) CSparseFeatures<INT>;
%template(SparseUIntFeatures) CSparseFeatures<UINT>;
%template(SparseLongFeatures) CSparseFeatures<LONG>;
%template(SparseUlongFeatures) CSparseFeatures<ULONG>;
%template(SparseRealFeatures) CSparseFeatures<DREAL>;
%template(SparseShortRealFeatures) CSparseFeatures<SHORTREAL>;
%template(SparseLongRealFeatures) CSparseFeatures<LONGREAL>;
