/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2009 Soeren Sonnenburg
 * Copyright (C) 2009 Fraunhofer Institute FIRST and Max-Planck-Society
 */

%rename(Cache) CCache;
%rename(ListElement) CListElement;
%rename(List) CList;
%rename(Signal) CSignal;
%rename(Time) CTime;
%rename(Hash) CHash;

%ignore RADIX_STACK_SIZE;
%ignore NUMTRAPPEDSIGS;
%ignore TRIE_TERMINAL_CHARACTER;
%ignore NO_CHILD;

#pragma SWIG nowarn=312,362,389
%warnfilter(509) CArray;
%warnfilter(509) CArray2;
%warnfilter(509) CArray3;

/* Templated Datatype Classes */
%include <shogun/lib/DataType.h>
namespace shogun
{
    %template(BoolSparseMatrix) SGSparseMatrix<bool>;
    %template(CharSparseMatrix) SGSparseMatrix<char>;
    %template(ByteSparseMatrix) SGSparseMatrix<uint8_t>;
    %template(WordSparseMatrix) SGSparseMatrix<uint16_t>;
    %template(ShortSparseMatrix) SGSparseMatrix<int16_t>;
    %template(IntSparseMatrix)  SGSparseMatrix<int32_t>;
    %template(UIntSparseMatrix)  SGSparseMatrix<uint32_t>;
    %template(LongIntSparseMatrix)  SGSparseMatrix<int64_t>;
    %template(ULongIntSparseMatrix)  SGSparseMatrix<uint64_t>;
    %template(ShortRealSparseMatrix) SGSparseMatrix<float32_t>;
    %template(RealSparseMatrix) SGSparseMatrix<float64_t>;
    %template(LongRealSparseMatrix) SGSparseMatrix<floatmax_t>;

    %template(BoolStringList) SGStringList<bool>;
    %template(CharStringList) SGStringList<char>;
    %template(ByteStringList) SGStringList<uint8_t>;
    %template(WordStringList) SGStringList<uint16_t>;
    %template(ShortStringList) SGStringList<int16_t>;
    %template(IntStringList)  SGStringList<int32_t>;
    %template(UIntStringList)  SGStringList<uint32_t>;
    %template(LongIntStringList)  SGStringList<int64_t>;
    %template(ULongIntStringList)  SGStringList<uint64_t>;
    %template(ShortRealStringList) SGStringList<float32_t>;
    %template(RealStringList) SGStringList<float64_t>;
    %template(LongRealStringList) SGStringList<floatmax_t>;


    %template(BoolString) SGString<bool>;
    %template(CharString) SGString<char>;
    %template(ByteString) SGString<uint8_t>;
    %template(WordString) SGString<uint16_t>;
    %template(ShortString) SGString<int16_t>;
    %template(IntString)  SGString<int32_t>;
    %template(UIntString)  SGString<uint32_t>;
    %template(LongIntString)  SGString<int64_t>;
    %template(ULongIntString)  SGString<uint64_t>;
    %template(ShortRealString) SGString<float32_t>;
    %template(RealString) SGString<float64_t>;
    %template(LongRealString) SGString<floatmax_t>;

    %template(BoolVector) SGVector<bool>;
    %template(CharVector) SGVector<char>;
    %template(ByteVector) SGVector<uint8_t>;
    %template(WordVector) SGVector<uint16_t>;
    %template(ShortVector) SGVector<int16_t>;
    %template(IntVector)  SGVector<int32_t>;
    %template(UIntVector)  SGVector<uint32_t>;
    %template(LongIntVector)  SGVector<int64_t>;
    %template(ULongIntVector)  SGVector<uint64_t>;
    %template(ShortRealVector) SGVector<float32_t>;
    %template(RealVector) SGVector<float64_t>;
    %template(LongRealVector) SGVector<floatmax_t>;

    %template(BoolMatrix) SGMatrix<bool>;
    %template(CharMatrix) SGMatrix<char>;
    %template(ByteMatrix) SGMatrix<uint8_t>;
    %template(WordMatrix) SGMatrix<uint16_t>;
    %template(ShortMatrix) SGMatrix<int16_t>;
    %template(IntMatrix)  SGMatrix<int32_t>;
    %template(UIntMatrix)  SGMatrix<uint32_t>;
    %template(LongIntMatrix)  SGMatrix<int64_t>;
    %template(ULongIntMatrix)  SGMatrix<uint64_t>;
    %template(ShortRealMatrix) SGMatrix<float32_t>;
    %template(RealMatrix) SGMatrix<float64_t>;
    %template(LongRealMatrix) SGMatrix<floatmax_t>;

    %template(BoolNDArray) SGNDArray<bool>;
    %template(CharNDArray) SGNDArray<char>;
    %template(ByteNDArray) SGNDArray<uint8_t>;
    %template(WordNDArray) SGNDArray<uint16_t>;
    %template(ShortNDArray) SGNDArray<int16_t>;
    %template(IntNDArray)  SGNDArray<int32_t>;
    %template(UIntNDArray)  SGNDArray<uint32_t>;
    %template(LongIntNDArray)  SGNDArray<int64_t>;
    %template(ULongIntNDArray)  SGNDArray<uint64_t>;
    %template(ShortRealNDArray) SGNDArray<float32_t>;
    %template(RealNDArray) SGNDArray<float64_t>;
    %template(LongRealNDArray) SGNDArray<floatmax_t>;
}

/* Include Class Headers to make them visible from within the target language */
/* Template Class DynamicArray */
%include <shogun/lib/DynamicArray.h>
%include <shogun/base/DynArray.h>

namespace shogun
{
    %template(DynamicCharArray) CDynamicArray<char>;
    %template(DynamicByteArray) CDynamicArray<uint8_t>;
    %template(DynamicShortArray) CDynamicArray<int16_t>;
    %template(DynamicWordArray) CDynamicArray<uint16_t>;
    %template(DynamicIntArray) CDynamicArray<int32_t>;
    %template(DynamicUIntArray) CDynamicArray<uint32_t>;
    %template(DynamicLongArray) CDynamicArray<int64_t>;
    %template(DynamicULongArray) CDynamicArray<uint64_t>;
    %template(DynamicShortRealArray) CDynamicArray<float32_t>;
    %template(DynamicRealArray) CDynamicArray<float64_t>;
    %template(DynamicPlifArray) DynArray<shogun::CPlifBase*>;
}

/* Template Class GCArray */
%include <shogun/lib/GCArray.h>
namespace shogun
{
    %template(PlifGCArray) CGCArray<shogun::CPlifBase*>;
}

/* Hash */
%include <shogun/lib/Hash.h>

/* Template Class Array */
/*%include <shogun/lib/Array.h>
namespace shogun
{
    %template(CharArray) CArray<char>;
    %template(ByteArray) CArray<uint8_t>;
    %template(ShortArray) CArray<int16_t>;
    %template(WordArray) CArray<uint16_t>;
    %template(IntArray) CArray<int32_t>;
    %template(UIntArray) CArray<uint32_t>;
    %template(LongArray) CArray<int64_t>;
    %template(ULongArray) CArray<uint64_t>;
    %template(ShortRealArray) CArray<float32_t>;
    %template(RealArray) CArray<float64_t>;
}*/

/* Template Class Array2 */
/*%include <shogun/lib/Array2.h>
namespace shogun
{
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
}*/

/* Template Class Array3 */
/*%include <shogun/lib/Array3.h>
namespace shogun
{
    %template(CharArray3) CArray3<char>;
    %template(ByteArray3) CArray3<uint8_t>;
    %template(ShortArray3) CArray3<int16_t>;
    %template(WordArray3) CArray3<uint16_t>;
    %template(IntArray3) CArray3<int32_t>;
    %template(UIntArray3) CArray3<uint32_t>;
    %template(LongArray3) CArray3<int64_t>;
    %template(ULongArray3) CArray3<uint64_t>;
    %template(ShortRealArray3) CArray3<float32_t>;
    %template(RealArray3) CArray3<float64_t>;
}*/

%include <shogun/lib/Cache.h>
%include <shogun/lib/GCArray.h>
%include <shogun/lib/List.h>
%include <shogun/lib/Signal.h>
%include <shogun/lib/Time.h>
%include <shogun/lib/Trie.h>
%include <shogun/lib/Compressor.h>
