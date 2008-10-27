%{
 #include "lib/DynamicArray.h" 
 #include "structure/PlifBase.h"
%}

%include "lib/DynamicArray.h"
%include "structure/PlifBase.i"

%template(DynamicCharArray) CDynamicArray<char>;
%template(DynamicByteArray) CDynamicArray<uint8_t>;
%template(DynamicShortArray) CDynamicArray<int16_t>;
%template(DynamicWordArray) CDynamicArray<uint16_t>;
%template(DynamicIntArray) CDynamicArray<int32_t>;
%template(DynamicUIntArray) CDynamicArray<uint32_t>;
%template(DynamicLongArray) CDynamicArray<int64_t>;
%template(DynamicULongArray) CDynamicArray<uint64_t>;
%template(DynamicShortRealArray) CDynamicArray<float32_t>;
%template(DynamicRealArray) CDynamicArray<DREAL>;
%template(DynamicPlifArray) CDynamicArray<CPlifBase*>;
