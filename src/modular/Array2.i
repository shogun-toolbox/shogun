%warnfilter(509) CArray2;
%{
 #include <shogun/lib/Array2.h>
%}

%include <shogun/lib/Array2.h>

%template(CharArray2) CArray2<char>;
%template(ByteArray2) CArray2<uint8_t>;
%template(ShortArray2) CArray2<int16_t>;
%template(WordArray2) CArray2<uint16_t>;
%template(IntArray2) CArray2<int32_t>;
%template(UIntArray2) CArray2<uint32_t>;
%template(LongArray2) CArray2<int64_t>;
%template(ULongArray2) CArray2<uint64_t>;
%template(ShortRealArray2) CArray2<float32_t>;
%template(RealArray2) CArray2<float64_t>;
