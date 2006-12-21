%{
 #include "lib/Array3.h" 
%}

%include "lib/Array3.h"

%template(CharArray3) CArray3<CHAR>;
%template(ByteArray3) CArray3<BYTE>;
%template(ShortArray3) CArray3<SHORT>;
%template(WordArray3) CArray3<WORD>;
%template(IntArray3) CArray3<INT>;
%template(UIntArray3) CArray3<UINT>;
%template(LongArray3) CArray3<LONG>;
%template(ULongArray3) CArray3<ULONG>;
%template(ShortRealArray3) CArray3<SHORTREAL>;
%template(RealArray3) CArray3<DREAL>;
