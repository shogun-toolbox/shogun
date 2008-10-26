%warnfilter(509) CArray3;
%{
 #include "lib/Array3.h" 
%}

%include "lib/Array3.h"

%template(CharArray3) CArray3<char>;
%template(ByteArray3) CArray3<uint8_t>;
%template(ShortArray3) CArray3<int16_t>;
%template(WordArray3) CArray3<uint16_t>;
%template(IntArray3) CArray3<int32_t>;
%template(UIntArray3) CArray3<uint32_t>;
%template(LongArray3) CArray3<int64_t>;
%template(ULongArray3) CArray3<uint64_t>;
%template(ShortRealArray3) CArray3<SHORTREAL>;
%template(RealArray3) CArray3<DREAL>;
