/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2009 Soeren Sonnenburg
 * Written (W) 2013 Heiko Strathmann
 * Copyright (C) 2009 Fraunhofer Institute FIRST and Max-Planck-Society
 * Copyright (c) 2012 Evgeniy Andreev (gsomix)
 */

#ifndef SWIGPYTHON
#define PROTOCOLS_SGVECTOR(class_name, type_name, format_str, typecode)
#endif

%rename(Cache) CCache;
%rename(ListElement) CListElement;
%rename(List) CList;
%rename(Signal) CSignal;
%rename(Time) CTime;
%rename(Hash) CHash;
%rename(StructuredData) StructuredData;
%rename(DynamicObjectArray) CDynamicObjectArray;
%rename(Tokenizer) CTokenizer;
%rename(DelimiterTokenizer) CDelimiterTokenizer;
%rename(NGramTokenizer) CNGramTokenizer;

%rename(IndexBlock) CIndexBlock;
%rename(IndexBlockRelation) CIndexBlockRelation;
%rename(IndexBlockGroup) CIndexBlockGroup;
%rename(IndexBlockTree) CIndexBlockTree;
%rename(Data) CData;

%rename(IndependentComputationEngine) CIndependentComputationEngine;
%rename(SerialComputationEngine) CSerialComputationEngine;


%ignore RADIX_STACK_SIZE;
%ignore NUMTRAPPEDSIGS;
%ignore TRIE_TERMINAL_CHARACTER;
%ignore NO_CHILD;

/* Templated Datatype Classes */
%include <shogun/lib/DataType.h>
%include <shogun/lib/SGReferencedData.h>
%include <shogun/lib/SGVector.h>
%include <shogun/lib/SGMatrix.h>
%include <shogun/lib/SGSparseVector.h>
%include <shogun/lib/SGSparseMatrix.h>
%include <shogun/lib/SGString.h>
%include <shogun/lib/SGStringList.h>
%include <shogun/lib/SGNDArray.h>
namespace shogun
{
    SERIALIZABLE_DUMMY(SGReferencedData);

#ifdef USE_BOOL
    %template(BoolSparseVector) SGSparseVector<bool>;
    SERIALIZABLE_DUMMY(SGSparseVector<bool>);
#endif
#ifdef USE_CHAR
    %template(CharSparseVector) SGSparseVector<char>;
    SERIALIZABLE_DUMMY(SGSparseVector<char>);
#endif
#ifdef USE_UINT8
    %template(ByteSparseVector) SGSparseVector<uint8_t>;
    SERIALIZABLE_DUMMY(SGSparseVector<uint8_t>);
#endif
#ifdef USE_UINT16
    %template(WordSparseVector) SGSparseVector<uint16_t>;
    SERIALIZABLE_DUMMY(SGSparseVector<uint16_t>);
#endif
#ifdef USE_INT16
    %template(ShortSparseVector) SGSparseVector<int16_t>;
    SERIALIZABLE_DUMMY(SGSparseVector<int16_t>);
#endif
#ifdef USE_INT32
    %template(IntSparseVector)  SGSparseVector<int32_t>;
    SERIALIZABLE_DUMMY(SGSparseVector<int32_t>);
#endif
#ifdef USE_UINT32
    %template(UIntSparseVector)  SGSparseVector<uint32_t>;
    SERIALIZABLE_DUMMY(SGSparseVector<uint32_t>);
#endif
#ifdef USE_INT64
    %template(LongIntSparseVector)  SGSparseVector<int64_t>;
    SERIALIZABLE_DUMMY(SGSparseVector<int64_t>);
#endif
#ifdef USE_UINT64
    %template(ULongIntSparseVector)  SGSparseVector<uint64_t>;
    SERIALIZABLE_DUMMY(SGSparseVector<uint64_t>);
#endif
#ifdef USE_FLOAT32
    %template(ShortRealSparseVector) SGSparseVector<float32_t>;
    SERIALIZABLE_DUMMY(SGSparseVector<float32_t>);
#endif
#ifdef USE_FLOAT64
    %template(RealSparseVector) SGSparseVector<float64_t>;
    SERIALIZABLE_DUMMY(SGSparseVector<float64_t>);
#endif
#ifdef USE_FLOATMAX
    %template(LongRealSparseVector) SGSparseVector<floatmax_t>;
    SERIALIZABLE_DUMMY(SGSparseVector<floatmax_t>);
#endif
#ifdef USE_COMPLEX128
    %template(ComplexSparseVector) SGSparseVector<complex128_t>;
    SERIALIZABLE_DUMMY(SGSparseVector<complex128_t>);
#endif
#ifdef USE_BOOL
    %template(BoolSparseMatrix) SGSparseMatrix<bool>;
    SERIALIZABLE_DUMMY(SGSparseMatrix<bool>);
#endif
#ifdef USE_CHAR
    %template(CharSparseMatrix) SGSparseMatrix<char>;
    SERIALIZABLE_DUMMY(SGSparseMatrix<char>);
#endif
#ifdef USE_UINT8
    %template(ByteSparseMatrix) SGSparseMatrix<uint8_t>;
    SERIALIZABLE_DUMMY(SGSparseMatrix<uint8_t>);
#endif
#ifdef USE_UINT16
    %template(WordSparseMatrix) SGSparseMatrix<uint16_t>;
    SERIALIZABLE_DUMMY(SGSparseMatrix<uint16_t>);
#endif
#ifdef USE_INT16
    %template(ShortSparseMatrix) SGSparseMatrix<int16_t>;
    SERIALIZABLE_DUMMY(SGSparseMatrix<int16_t>);
#endif
#ifdef USE_INT32
    %template(IntSparseMatrix)  SGSparseMatrix<int32_t>;
    SERIALIZABLE_DUMMY(SGSparseMatrix<int32_t>);
#endif
#ifdef USE_UINT32
    %template(UIntSparseMatrix)  SGSparseMatrix<uint32_t>;
    SERIALIZABLE_DUMMY(SGSparseMatrix<uint32_t>);
#endif
#ifdef USE_INT64
    %template(LongIntSparseMatrix)  SGSparseMatrix<int64_t>;
    SERIALIZABLE_DUMMY(SGSparseMatrix<int64_t>);
#endif
#ifdef USE_UINT64
    %template(ULongIntSparseMatrix)  SGSparseMatrix<uint64_t>;
    SERIALIZABLE_DUMMY(SGSparseMatrix<uint64_t>);
#endif
#ifdef USE_FLOAT32
    %template(ShortRealSparseMatrix) SGSparseMatrix<float32_t>;
    SERIALIZABLE_DUMMY(SGSparseMatrix<float32_t>);
#endif
#ifdef USE_FLOAT64
    %template(RealSparseMatrix) SGSparseMatrix<float64_t>;
    SERIALIZABLE_DUMMY(SGSparseMatrix<float64_t>);
#endif
#ifdef USE_FLOATMAX
    %template(LongRealSparseMatrix) SGSparseMatrix<floatmax_t>;
    SERIALIZABLE_DUMMY(SGSparseMatrix<floatmax_t>);
#endif
#ifdef USE_COMPLEX128
    %template(ComplexSparseMatrix) SGSparseMatrix<complex128_t>;
    SERIALIZABLE_DUMMY(SGSparseMatrix<complex128_t>);
#endif

#ifdef USE_BOOL
    %template(BoolStringList) SGStringList<bool>;
    SERIALIZABLE_DUMMY(SGStringList<bool>);
#endif
#ifdef USE_CHAR
    %template(CharStringList) SGStringList<char>;
    SERIALIZABLE_DUMMY(SGStringList<char>);
#endif
#ifdef USE_UINT8
    %template(ByteStringList) SGStringList<uint8_t>;
    SERIALIZABLE_DUMMY(SGStringList<uint8_t>);
#endif
#ifdef USE_UINT16
    %template(WordStringList) SGStringList<uint16_t>;
    SERIALIZABLE_DUMMY(SGStringList<uint16_t>);
#endif
#ifdef USE_INT16
    %template(ShortStringList) SGStringList<int16_t>;
    SERIALIZABLE_DUMMY(SGStringList<int16_t>);
#endif
#ifdef USE_INT32
    %template(IntStringList)  SGStringList<int32_t>;
    SERIALIZABLE_DUMMY(SGStringList<int32_t>);
#endif
#ifdef USE_UINT32
    %template(UIntStringList)  SGStringList<uint32_t>;
    SERIALIZABLE_DUMMY(SGStringList<uint32_t>);
#endif
#ifdef USE_INT64
    %template(LongIntStringList)  SGStringList<int64_t>;
    SERIALIZABLE_DUMMY(SGStringList<int64_t>);
#endif
#ifdef USE_UINT64
    %template(ULongIntStringList)  SGStringList<uint64_t>;
    SERIALIZABLE_DUMMY(SGStringList<uint64_t>);
#endif
#ifdef USE_FLOAT32
    %template(ShortRealStringList) SGStringList<float32_t>;
    SERIALIZABLE_DUMMY(SGStringList<float32_t>);
#endif
#ifdef USE_FLOAT64
    %template(RealStringList) SGStringList<float64_t>;
    SERIALIZABLE_DUMMY(SGStringList<float64_t>);
#endif
#ifdef USE_FLOATMAX
    %template(LongRealStringList) SGStringList<floatmax_t>;
    SERIALIZABLE_DUMMY(SGStringList<floatmax_t>);
#endif

#ifdef USE_BOOL
    %template(BoolString) SGString<bool>;
    SERIALIZABLE_DUMMY(SGString<bool>);
#endif
#ifdef USE_CHAR
    %template(CharString) SGString<char>;
    SERIALIZABLE_DUMMY(SGString<char>);
#endif
#ifdef USE_UINT8
    %template(ByteString) SGString<uint8_t>;
    SERIALIZABLE_DUMMY(SGString<uint8_t>);
#endif
#ifdef USE_UINT16
    %template(WordString) SGString<uint16_t>;
    SERIALIZABLE_DUMMY(SGString<uint16_t>);
#endif
#ifdef USE_INT16
    %template(ShortString) SGString<int16_t>;
    SERIALIZABLE_DUMMY(SGString<int16_t>);
#endif
#ifdef USE_INT32
    %template(IntString)  SGString<int32_t>;
    SERIALIZABLE_DUMMY(SGString<int32_t>);
#endif
#ifdef USE_UINT32
    %template(UIntString)  SGString<uint32_t>;
    SERIALIZABLE_DUMMY(SGString<uint32_t>);
#endif
#ifdef USE_INT64
    %template(LongIntString)  SGString<int64_t>;
    SERIALIZABLE_DUMMY(SGString<int64_t>);
#endif
#ifdef USE_UINT64
    %template(ULongIntString)  SGString<uint64_t>;
    SERIALIZABLE_DUMMY(SGString<uint64_t>);
#endif
#ifdef USE_FLOAT32
    %template(ShortRealString) SGString<float32_t>;
    SERIALIZABLE_DUMMY(SGString<float32_t>);
#endif
#ifdef USE_FLOAT64
    %template(RealString) SGString<float64_t>;
    SERIALIZABLE_DUMMY(SGString<float64_t>);
#endif
#ifdef USE_FLOATMAX
    %template(LongRealString) SGString<floatmax_t>;
    SERIALIZABLE_DUMMY(SGString<floatmax_t>);
#endif

#if !defined(SWIGPERL)
%ignore SGVector<float64_t>::operator+=;
%ignore SGVector<float64_t>::operator+;
#endif

#ifdef USE_BOOL
    PROTOCOLS_SGVECTOR(BoolVector, bool, "?\0", NPY_BOOL)
    %template(BoolVector) SGVector<bool>;
    SERIALIZABLE_DUMMY(SGVector<bool>);
#endif
#ifdef USE_CHAR
    PROTOCOLS_SGVECTOR(CharVector, char, "c\0", NPY_STRING)
    %template(CharVector) SGVector<char>;
    SERIALIZABLE_DUMMY(SGVector<char>);
#endif
#ifdef USE_UINT8
    PROTOCOLS_SGVECTOR(ByteVector, uint8_t, "B\0", NPY_UINT8)
    %template(ByteVector) SGVector<uint8_t>;
    SERIALIZABLE_DUMMY(SGVector<uint8_t>);
#endif
#ifdef USE_UINT16
    PROTOCOLS_SGVECTOR(WordVector, uint16_t, "H\0", NPY_UINT16)
    %template(WordVector) SGVector<uint16_t>;
    SERIALIZABLE_DUMMY(SGVector<uint16_t>);
#endif
#ifdef USE_INT16
    PROTOCOLS_SGVECTOR(ShortVector, int16_t, "h\0", NPY_INT16)
    %template(ShortVector) SGVector<int16_t>;
    SERIALIZABLE_DUMMY(SGVector<int16_t>);
#endif
#ifdef USE_INT32
    PROTOCOLS_SGVECTOR(IntVector, int32_t, "i\0", NPY_INT32)
    %template(IntVector)  SGVector<int32_t>;
    SERIALIZABLE_DUMMY(SGVector<int32_t>);
#endif
#ifdef USE_UINT32
    PROTOCOLS_SGVECTOR(UIntVector, uint32_t, "I\0", NPY_UINT32)
    %template(UIntVector)  SGVector<uint32_t>;
    SERIALIZABLE_DUMMY(SGVector<uint32_t>);
#endif
#ifdef USE_INT64
    PROTOCOLS_SGVECTOR(LongIntVector, int64_t, "l\0", NPY_INT64)
    %template(LongIntVector)  SGVector<int64_t>;
    SERIALIZABLE_DUMMY(SGVector<int64_t>);
#endif
#ifdef USE_UINT64
    PROTOCOLS_SGVECTOR(ULongIntVector, int64_t, "L\0", NPY_UINT64)
    %template(ULongIntVector)  SGVector<uint64_t>;
    SERIALIZABLE_DUMMY(SGVector<uint64_t>);
#endif
#ifdef USE_FLOAT32
    PROTOCOLS_SGVECTOR(ShortRealVector, float32_t, "f\0", NPY_FLOAT32)
    %template(ShortRealVector) SGVector<float32_t>;
    SERIALIZABLE_DUMMY(SGVector<float32_t>);
#endif
#ifdef USE_FLOAT64
    PROTOCOLS_SGVECTOR(RealVector, float64_t, "d\0", NPY_FLOAT64)
    %template(RealVector) SGVector<float64_t>;
    SERIALIZABLE_DUMMY(SGVector<float64_t>);
#endif
#ifdef USE_FLOATMAX
    %template(LongRealVector) SGVector<floatmax_t>;
    SERIALIZABLE_DUMMY(SGVector<floatmax_t>);
#endif
#ifdef USE_COMPLEX128
    PROTOCOLS_SGVECTOR(ComplexVector, complex128_t, "d\0", NPY_CDOUBLE)
    %template(ComplexVector) SGVector<complex128_t>;
    SERIALIZABLE_DUMMY(SGVector<complex128_t>);
#endif

#ifdef USE_BOOL
    %template(BoolMatrix) SGMatrix<bool>;
    SERIALIZABLE_DUMMY(SGMatrix<bool>);
#endif
#ifdef USE_CHAR
    %template(CharMatrix) SGMatrix<char>;
    SERIALIZABLE_DUMMY(SGMatrix<char>);
#endif
#ifdef USE_UINT8
    %template(ByteMatrix) SGMatrix<uint8_t>;
    SERIALIZABLE_DUMMY(SGMatrix<uint8_t>);
#endif
#ifdef USE_UINT16
    %template(WordMatrix) SGMatrix<uint16_t>;
    SERIALIZABLE_DUMMY(SGMatrix<uint16_t>);
#endif
#ifdef USE_INT16
    %template(ShortMatrix) SGMatrix<int16_t>;
    SERIALIZABLE_DUMMY(SGMatrix<int16_t>);
#endif
#ifdef USE_INT32
    %template(IntMatrix)  SGMatrix<int32_t>;
    SERIALIZABLE_DUMMY(SGMatrix<int32_t>);
#endif
#ifdef USE_UINT32
    %template(UIntMatrix)  SGMatrix<uint32_t>;
    SERIALIZABLE_DUMMY(SGMatrix<uint32_t>);
#endif
#ifdef USE_INT64
    %template(LongIntMatrix)  SGMatrix<int64_t>;
    SERIALIZABLE_DUMMY(SGMatrix<int64_t>);
#endif
#ifdef USE_UINT64
    %template(ULongIntMatrix)  SGMatrix<uint64_t>;
    SERIALIZABLE_DUMMY(SGMatrix<uint64_t>);
#endif
#ifdef USE_FLOAT32
    %template(ShortRealMatrix) SGMatrix<float32_t>;
    SERIALIZABLE_DUMMY(SGMatrix<float32_t>);
#endif
#ifdef USE_FLOAT64
    %template(RealMatrix) SGMatrix<float64_t>;
    SERIALIZABLE_DUMMY(SGMatrix<float64_t>);
#endif
#ifdef USE_FLOATMAX
    %template(LongRealMatrix) SGMatrix<floatmax_t>;
    SERIALIZABLE_DUMMY(SGMatrix<floatmax_t>);
#endif
#ifdef USE_COMPLEX128
    %template(ComplexMatrix) SGMatrix<complex128_t>;
    SERIALIZABLE_DUMMY(SGMatrix<complex128_t>);
#endif

#ifdef USE_BOOL
    %template(BoolNDArray) SGNDArray<bool>;
    SERIALIZABLE_DUMMY(SGNDArray<bool>);
#endif
#ifdef USE_CHAR
    %template(CharNDArray) SGNDArray<char>;
    SERIALIZABLE_DUMMY(SGNDArray<char>);
#endif
#ifdef USE_UINT16
    %template(WordNDArray) SGNDArray<uint16_t>;
    SERIALIZABLE_DUMMY(SGNDArray<uint16_t>);
#endif
#ifdef USE_UINT8
    %template(ByteNDArray) SGNDArray<uint8_t>;
    SERIALIZABLE_DUMMY(SGNDArray<uint8_t>);
#endif
#ifdef USE_INT16
    %template(ShortNDArray) SGNDArray<int16_t>;
    SERIALIZABLE_DUMMY(SGNDArray<int16_t>);
#endif
#ifdef USE_INT32
    %template(IntNDArray)  SGNDArray<int32_t>;
    SERIALIZABLE_DUMMY(SGNDArray<int32_t>);
#endif
#ifdef USE_UINT32
    %template(UIntNDArray)  SGNDArray<uint32_t>;
    SERIALIZABLE_DUMMY(SGNDArray<uint32_t>);
#endif
#ifdef USE_INT64
    %template(LongIntNDArray)  SGNDArray<int64_t>;
    SERIALIZABLE_DUMMY(SGNDArray<int64_t>);
#endif
#ifdef USE_UINT64
    %template(ULongIntNDArray)  SGNDArray<uint64_t>;
    SERIALIZABLE_DUMMY(SGNDArray<uint64_t>);
#endif
#ifdef USE_FLOAT32
    %template(ShortRealNDArray) SGNDArray<float32_t>;
    SERIALIZABLE_DUMMY(SGNDArray<float32_t>);
#endif
#ifdef USE_FLOAT64
    %template(RealNDArray) SGNDArray<float64_t>;
    SERIALIZABLE_DUMMY(SGNDArray<float64_t>);
#endif
#ifdef USE_FLOATMAX
    %template(LongRealNDArray) SGNDArray<floatmax_t>;
    SERIALIZABLE_DUMMY(SGNDArray<floatmax_t>);
#endif
}


/* Include Class Headers to make them visible from within the target language */
/* Template Class DynamicArray */
%include <shogun/lib/DynamicArray.h>
%include <shogun/base/DynArray.h>
namespace shogun
{
#ifdef USE_CHAR
        %template(DynamicCharArray) CDynamicArray<char>;
        SERIALIZABLE_DUMMY(CDynamicArray<char>);
#endif
#ifdef USE_UINT8
        %template(DynamicByteArray) CDynamicArray<uint8_t>;
        SERIALIZABLE_DUMMY(CDynamicArray<uint8_t>);
#endif
#ifdef USE_INT16
        %template(DynamicShortArray) CDynamicArray<int16_t>;
        SERIALIZABLE_DUMMY(CDynamicArray<int16_t>);
#endif
#ifdef USE_UINT16
        %template(DynamicWordArray) CDynamicArray<uint16_t>;
        SERIALIZABLE_DUMMY(CDynamicArray<uint16_t>);
#endif
#ifdef USE_INT32
        %template(DynamicIntArray) CDynamicArray<int32_t>;
        SERIALIZABLE_DUMMY(CDynamicArray<int32_t>);
#endif
#ifdef USE_UINT32
        %template(DynamicUIntArray) CDynamicArray<uint32_t>;
        SERIALIZABLE_DUMMY(CDynamicArray<uint32_t>);
#endif
#ifdef USE_INT64
        %template(DynamicLongArray) CDynamicArray<int64_t>;
        SERIALIZABLE_DUMMY(CDynamicArray<int64_t>);
#endif
#ifdef USE_UINT64
        %template(DynamicULongArray) CDynamicArray<uint64_t>;
        SERIALIZABLE_DUMMY(CDynamicArray<uint64_t>);
#endif
#ifdef USE_FLOAT32
        %template(DynamicShortRealArray) CDynamicArray<float32_t>;
        SERIALIZABLE_DUMMY(CDynamicArray<float32_t>);
#endif
#ifdef USE_FLOAT64
        %template(DynamicRealArray) CDynamicArray<float64_t>;
        SERIALIZABLE_DUMMY(CDynamicArray<float64_t>);
#endif
        %template(DynamicPlifArray) DynArray<shogun::CPlifBase*>;
        SERIALIZABLE_DUMMY(DynArray<shogun::CPlifBase*>);
}
/* Template Class GCArray */
%include <shogun/lib/GCArray.h>
namespace shogun
{
        %template(PlifGCArray) CGCArray<shogun::CPlifBase*>;
        SERIALIZABLE_DUMMY(CGCArray<shogun::CPlifBase*>);
}

/* Hash */
%include <shogun/lib/Hash.h>

%include <shogun/lib/Tokenizer.h>
%include <shogun/lib/DelimiterTokenizer.h>
%include <shogun/lib/NGramTokenizer.h>
%include <shogun/lib/Cache.h>
%include <shogun/lib/List.h>
%include <shogun/lib/Signal.h>
%include <shogun/lib/Time.h>
%include <shogun/lib/Trie.h>
%include <shogun/lib/Compressor.h>
%include <shogun/lib/StructuredDataTypes.h>
%include <shogun/lib/StructuredData.h>
%include <shogun/lib/DynamicObjectArray.h>
%include <shogun/lib/IndexBlock.h>
%include <shogun/lib/IndexBlockRelation.h>
%include <shogun/lib/IndexBlockGroup.h>
%include <shogun/lib/IndexBlockTree.h>
%include <shogun/lib/Data.h>

/* Computation framework */

/* Computation Engine */
%rename (IndependentComputationEngine) CIndependentComputationEngine;
%rename (SerialComputationEngine) CSerialComputationEngine;

%include <shogun/lib/computation/engine/IndependentComputationEngine.h>
%include <shogun/lib/computation/engine/SerialComputationEngine.h>

/* Independent compution-job */
%rename (IndependentJob) CIndepenentJob;
%include <shogun/lib/computation/job/IndependentJob.h>

/* Independent computation-job results */
%rename (JobResult) CJobResult;
%include <shogun/lib/computation/jobresult/JobResult.h>
%include <shogun/lib/computation/jobresult/ScalarResult.h>
namespace shogun
{
#ifdef USE_CHAR
  %template(ScalarCharResult) CScalarResult<char>;
#endif
#ifdef USE_BOOL
  %template(ScalarBoolResult) CScalarResult<bool>;
#endif
#ifdef USE_UINT8
  %template(ScalarByteResult) CScalarResult<uint8_t>;
#endif
#ifdef USE_INT16
  %template(ScalarShortResult) CScalarResult<int16_t>;
#endif
#ifdef USE_UINT16
  %template(ScalarWordResult) CScalarResult<uint16_t>;
#endif
#ifdef USE_INT32
  %template(ScalarIntResult) CScalarResult<int32_t>;
#endif
#ifdef USE_UINT32
  %template(ScalarUIntResult) CScalarResult<uint32_t>;
#endif
#ifdef USE_INT64
  %template(ScalarLongResult) CScalarResult<int64_t>;
#endif
#ifdef USE_UINT64
  %template(ScalarULongResult) CScalarResult<uint64_t>;
#endif
#ifdef USE_FLOAT32
  %template(ScalarShortRealResult) CScalarResult<float32_t>;
#endif
#ifdef USE_FLOAT64
  %template(ScalarRealResult) CScalarResult<float64_t>;
#endif
}

%include <shogun/lib/computation/jobresult/VectorResult.h>
namespace shogun
{
#ifdef USE_CHAR
  %template(VectorCharResult) CVectorResult<char>;
#endif
#ifdef USE_BOOL
  %template(VectorBoolResult) CVectorResult<bool>;
#endif
#ifdef USE_UINT8
  %template(VectorByteResult) CVectorResult<uint8_t>;
#endif
#ifdef USE_INT16
  %template(VectorShortResult) CVectorResult<int16_t>;
#endif
#ifdef USE_UINT16
  %template(VectorWordResult) CVectorResult<uint16_t>;
#endif
#ifdef USE_INT32
  %template(VectorIntResult) CVectorResult<int32_t>;
#endif
#ifdef USE_UINT32
  %template(VectorUIntResult) CVectorResult<uint32_t>;
#endif
#ifdef USE_INT64
  %template(VectorLongResult) CVectorResult<int64_t>;
#endif
#ifdef USE_UINT64
  %template(VectorULongResult) CVectorResult<uint64_t>;
#endif
#ifdef USE_FLOAT32
  %template(VectorShortRealResult) CVectorResult<float32_t>;
#endif
#ifdef USE_FLOAT64
  %template(VectorRealResult) CVectorResult<float64_t>;
#endif
}
