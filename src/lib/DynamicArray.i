%{
 #include "lib/DynamicArray.h" 
 #include "structure/PlifBase.h"
%}

%include "lib/DynamicArray.h"
%include "structure/PlifBase.i"

%template(DynamicCharArray) CDynamicArray<CHAR>;
%template(DynamicByteArray) CDynamicArray<BYTE>;
%template(DynamicShortArray) CDynamicArray<SHORT>;
%template(DynamicWordArray) CDynamicArray<WORD>;
%template(DynamicIntArray) CDynamicArray<INT>;
%template(DynamicUIntArray) CDynamicArray<UINT>;
%template(DynamicLongArray) CDynamicArray<LONG>;
%template(DynamicULongArray) CDynamicArray<ULONG>;
%template(DynamicShortRealArray) CDynamicArray<SHORTREAL>;
%template(DynamicRealArray) CDynamicArray<DREAL>;
%template(DynamicPlifArray) CDynamicArray<CPlifBase*>;
