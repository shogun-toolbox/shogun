%warnfilter(509) CArray;
%{
 #include "lib/Array.h" 
%}

%include "lib/Array.h"

%template(CharArray) CArray<char>;
%template(ByteArray) CArray<uint8_t>;
%template(ShortArray) CArray<SHORT>;
%template(WordArray) CArray<uint16_t>;
%template(IntArray) CArray<int32_t>;
%template(UIntArray) CArray<uint32_t>;
%template(LongArray) CArray<int64_t>;
%template(ULongArray) CArray<uint64_t>;
%template(ShortRealArray) CArray<SHORTREAL>;
%template(RealArray) CArray<DREAL>;
