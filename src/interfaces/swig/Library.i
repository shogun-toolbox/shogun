/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Viktor Gal, Shubham Shukla, Sergey Lisitsyn
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
%rename(StructuredData) CStructuredData;
%rename(DynamicObjectArray) CDynamicObjectArray;
%rename(Tokenizer) CTokenizer;
%rename(DelimiterTokenizer) CDelimiterTokenizer;
%rename(NGramTokenizer) CNGramTokenizer;

%rename(IndexBlock) CIndexBlock;
%rename(IndexBlockRelation) CIndexBlockRelation;
%rename(IndexBlockGroup) CIndexBlockGroup;
%rename(IndexBlockTree) CIndexBlockTree;
%rename(Data) CData;

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
#ifdef USE_BOOL
    %template(BoolSparseVector) SGSparseVector<bool>;
#endif
#ifdef USE_CHAR
    %template(CharSparseVector) SGSparseVector<char>;
#endif
#ifdef USE_UINT8
    %template(ByteSparseVector) SGSparseVector<uint8_t>;
#endif
#ifdef USE_UINT16
    %template(WordSparseVector) SGSparseVector<uint16_t>;
#endif
#ifdef USE_INT16
    %template(ShortSparseVector) SGSparseVector<int16_t>;
#endif
#ifdef USE_INT32
    %template(IntSparseVector)  SGSparseVector<int32_t>;
#endif
#ifdef USE_UINT32
    %template(UIntSparseVector)  SGSparseVector<uint32_t>;
#endif
#ifdef USE_INT64
    %template(LongIntSparseVector)  SGSparseVector<int64_t>;
#endif
#ifdef USE_UINT64
    %template(ULongIntSparseVector)  SGSparseVector<uint64_t>;
#endif
#ifdef USE_FLOAT32
    %template(ShortRealSparseVector) SGSparseVector<float32_t>;
#endif
#ifdef USE_FLOAT64
    %template(RealSparseVector) SGSparseVector<float64_t>;
#endif
#ifdef USE_FLOATMAX
    %template(LongRealSparseVector) SGSparseVector<floatmax_t>;
#endif
#ifdef USE_COMPLEX128
    %template(ComplexSparseVector) SGSparseVector<complex128_t>;
#endif
#ifdef USE_BOOL
    %template(BoolSparseMatrix) SGSparseMatrix<bool>;
#endif
#ifdef USE_CHAR
    %template(CharSparseMatrix) SGSparseMatrix<char>;
#endif
#ifdef USE_UINT8
    %template(ByteSparseMatrix) SGSparseMatrix<uint8_t>;
#endif
#ifdef USE_UINT16
    %template(WordSparseMatrix) SGSparseMatrix<uint16_t>;
#endif
#ifdef USE_INT16
    %template(ShortSparseMatrix) SGSparseMatrix<int16_t>;
#endif
#ifdef USE_INT32
    %template(IntSparseMatrix)  SGSparseMatrix<int32_t>;
#endif
#ifdef USE_UINT32
    %template(UIntSparseMatrix)  SGSparseMatrix<uint32_t>;
#endif
#ifdef USE_INT64
    %template(LongIntSparseMatrix)  SGSparseMatrix<int64_t>;
#endif
#ifdef USE_UINT64
    %template(ULongIntSparseMatrix)  SGSparseMatrix<uint64_t>;
#endif
#ifdef USE_FLOAT32
    %template(ShortRealSparseMatrix) SGSparseMatrix<float32_t>;
#endif
#ifdef USE_FLOAT64
    %template(RealSparseMatrix) SGSparseMatrix<float64_t>;
#endif
#ifdef USE_FLOATMAX
    %template(LongRealSparseMatrix) SGSparseMatrix<floatmax_t>;
#endif
#ifdef USE_COMPLEX128
    %template(ComplexSparseMatrix) SGSparseMatrix<complex128_t>;
#endif

#ifdef USE_BOOL
    %template(BoolStringList) SGStringList<bool>;
#endif
#ifdef USE_CHAR
    %template(CharStringList) SGStringList<char>;
#endif
#ifdef USE_UINT8
    %template(ByteStringList) SGStringList<uint8_t>;
#endif
#ifdef USE_UINT16
    %template(WordStringList) SGStringList<uint16_t>;
#endif
#ifdef USE_INT16
    %template(ShortStringList) SGStringList<int16_t>;
#endif
#ifdef USE_INT32
    %template(IntStringList)  SGStringList<int32_t>;
#endif
#ifdef USE_UINT32
    %template(UIntStringList)  SGStringList<uint32_t>;
#endif
#ifdef USE_INT64
    %template(LongIntStringList)  SGStringList<int64_t>;
#endif
#ifdef USE_UINT64
    %template(ULongIntStringList)  SGStringList<uint64_t>;
#endif
#ifdef USE_FLOAT32
    %template(ShortRealStringList) SGStringList<float32_t>;
#endif
#ifdef USE_FLOAT64
    %template(RealStringList) SGStringList<float64_t>;
#endif
#ifdef USE_FLOATMAX
    %template(LongRealStringList) SGStringList<floatmax_t>;
#endif

#ifdef USE_BOOL
    %template(BoolString) SGString<bool>;
#endif
#ifdef USE_CHAR
    %template(CharString) SGString<char>;
#endif
#ifdef USE_UINT8
    %template(ByteString) SGString<uint8_t>;
#endif
#ifdef USE_UINT16
    %template(WordString) SGString<uint16_t>;
#endif
#ifdef USE_INT16
    %template(ShortString) SGString<int16_t>;
#endif
#ifdef USE_INT32
    %template(IntString)  SGString<int32_t>;
#endif
#ifdef USE_UINT32
    %template(UIntString)  SGString<uint32_t>;
#endif
#ifdef USE_INT64
    %template(LongIntString)  SGString<int64_t>;
#endif
#ifdef USE_UINT64
    %template(ULongIntString)  SGString<uint64_t>;
#endif
#ifdef USE_FLOAT32
    %template(ShortRealString) SGString<float32_t>;
#endif
#ifdef USE_FLOAT64
    %template(RealString) SGString<float64_t>;
#endif
#ifdef USE_FLOATMAX
    %template(LongRealString) SGString<floatmax_t>;
#endif

#if !defined(SWIGPERL)
%ignore SGVector<float64_t>::operator+=;
%ignore SGVector<float64_t>::operator+;
#endif

#ifdef USE_BOOL
    PROTOCOLS_SGVECTOR(BoolVector, bool, "?\0", NPY_BOOL)
    %template(BoolVector) SGVector<bool>;
#endif
#ifdef USE_CHAR
    PROTOCOLS_SGVECTOR(CharVector, char, "c\0", NPY_STRING)
    %template(CharVector) SGVector<char>;
#endif
#ifdef USE_UINT8
    PROTOCOLS_SGVECTOR(ByteVector, uint8_t, "B\0", NPY_UINT8)
    %template(ByteVector) SGVector<uint8_t>;
#endif
#ifdef USE_UINT16
    PROTOCOLS_SGVECTOR(WordVector, uint16_t, "H\0", NPY_UINT16)
    %template(WordVector) SGVector<uint16_t>;
#endif
#ifdef USE_INT16
    PROTOCOLS_SGVECTOR(ShortVector, int16_t, "h\0", NPY_INT16)
    %template(ShortVector) SGVector<int16_t>;
#endif
#ifdef USE_INT32
    PROTOCOLS_SGVECTOR(IntVector, int32_t, "i\0", NPY_INT32)
    %template(IntVector)  SGVector<int32_t>;
#endif
#ifdef USE_INT64
    PROTOCOLS_SGVECTOR(LongIntVector, int64_t, "l\0", NPY_INT64)
    %template(LongIntVector)  SGVector<int64_t>;
#endif
#ifdef USE_UINT64
    PROTOCOLS_SGVECTOR(ULongIntVector, int64_t, "L\0", NPY_UINT64)
    %template(ULongIntVector)  SGVector<uint64_t>;
#endif
#ifdef USE_FLOAT32
    PROTOCOLS_SGVECTOR(ShortRealVector, float32_t, "f\0", NPY_FLOAT32)
    %template(ShortRealVector) SGVector<float32_t>;
#endif
#ifdef USE_FLOAT64
    PROTOCOLS_SGVECTOR(RealVector, float64_t, "d\0", NPY_FLOAT64)
    %template(RealVector) SGVector<float64_t>;
#endif
#ifdef USE_FLOATMAX
    %template(LongRealVector) SGVector<floatmax_t>;
#endif
#ifdef USE_COMPLEX128
    PROTOCOLS_SGVECTOR(ComplexVector, complex128_t, "d\0", NPY_CDOUBLE)
    %template(ComplexVector) SGVector<complex128_t>;
#endif

#ifdef USE_BOOL
    %template(BoolMatrix) SGMatrix<bool>;
#endif
#ifdef USE_CHAR
    %template(CharMatrix) SGMatrix<char>;
#endif
#ifdef USE_UINT8
    %template(ByteMatrix) SGMatrix<uint8_t>;
#endif
#ifdef USE_UINT16
    %template(WordMatrix) SGMatrix<uint16_t>;
#endif
#ifdef USE_INT16
    %template(ShortMatrix) SGMatrix<int16_t>;
#endif
#ifdef USE_INT32
    %template(IntMatrix)  SGMatrix<int32_t>;
#endif
#ifdef USE_UINT32
    %template(UIntMatrix)  SGMatrix<uint32_t>;
#endif
#ifdef USE_INT64
    %template(LongIntMatrix)  SGMatrix<int64_t>;
#endif
#ifdef USE_UINT64
    %template(ULongIntMatrix)  SGMatrix<uint64_t>;
#endif
#ifdef USE_FLOAT32
    %template(ShortRealMatrix) SGMatrix<float32_t>;
#endif
#ifdef USE_FLOAT64
    %template(RealMatrix) SGMatrix<float64_t>;
#endif
#ifdef USE_FLOATMAX
    %template(LongRealMatrix) SGMatrix<floatmax_t>;
#endif
#ifdef USE_COMPLEX128
    %template(ComplexMatrix) SGMatrix<complex128_t>;
#endif

#ifdef USE_BOOL
    %template(BoolNDArray) SGNDArray<bool>;
#endif
#ifdef USE_CHAR
    %template(CharNDArray) SGNDArray<char>;
#endif
#ifdef USE_UINT16
    %template(WordNDArray) SGNDArray<uint16_t>;
#endif
#ifdef USE_UINT8
    %template(ByteNDArray) SGNDArray<uint8_t>;
#endif
#ifdef USE_INT16
    %template(ShortNDArray) SGNDArray<int16_t>;
#endif
#ifdef USE_INT32
    %template(IntNDArray)  SGNDArray<int32_t>;
#endif
#ifdef USE_UINT32
    %template(UIntNDArray)  SGNDArray<uint32_t>;
#endif
#ifdef USE_INT64
    %template(LongIntNDArray)  SGNDArray<int64_t>;
#endif
#ifdef USE_UINT64
    %template(ULongIntNDArray)  SGNDArray<uint64_t>;
#endif
#ifdef USE_FLOAT32
    %template(ShortRealNDArray) SGNDArray<float32_t>;
#endif
#ifdef USE_FLOAT64
    %template(RealNDArray) SGNDArray<float64_t>;
#endif
#ifdef USE_FLOATMAX
    %template(LongRealNDArray) SGNDArray<floatmax_t>;
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
#endif
#ifdef USE_UINT8
        %template(DynamicByteArray) CDynamicArray<uint8_t>;
#endif
#ifdef USE_INT16
        %template(DynamicShortArray) CDynamicArray<int16_t>;
#endif
#ifdef USE_UINT16
        %template(DynamicWordArray) CDynamicArray<uint16_t>;
#endif
#ifdef USE_INT32
        %template(DynamicIntArray) CDynamicArray<int32_t>;
#endif
#ifdef USE_UINT32
        %template(DynamicUIntArray) CDynamicArray<uint32_t>;
#endif
#ifdef USE_INT64
        %template(DynamicLongArray) CDynamicArray<int64_t>;
#endif
#ifdef USE_UINT64
        %template(DynamicULongArray) CDynamicArray<uint64_t>;
#endif
#ifdef USE_FLOAT32
        %template(DynamicShortRealArray) CDynamicArray<float32_t>;
#endif
#ifdef USE_FLOAT64
        %template(DynamicRealArray) CDynamicArray<float64_t>;
#endif
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
namespace shogun
{
%extend CDynamicObjectArray
{
    template <typename T, typename X = typename std::enable_if_t<std::is_same<SGVector<typename extract_value_type<T>::value_type>, T>::value>>
    void append_element_vector(T v, const char* name="")
    {
        $self->append_element(v, name);
    }

    template <typename T, typename X = typename std::enable_if_t<std::is_same<SGMatrix<typename extract_value_type<T>::value_type>, T>::value>>
    void append_element_matrix(T v, const char* name="")
    {
        $self->append_element(v, name);
    }

    template <typename T, typename X = typename std::enable_if_t<std::is_same<SGStringList<typename extract_value_type<T>::value_type>, T>::value>>
    void append_element_string_list(T v, const char* name="")
    {
        $self->append_element(v, name);
    }
}

    /* Specialize DynamicObjectArray::append_element function */
#ifdef USE_FLOAT64
    %template(append_element_real_vector) CDynamicObjectArray::append_element_vector<SGVector<float64_t>, SGVector<float64_t>>;
    %template(append_element_real_matrix) CDynamicObjectArray::append_element_matrix<SGMatrix<float64_t>, SGMatrix<float64_t>>;
    %template(append_element_real) CDynamicObjectArray::append_element<float64_t, float64_t>;
#ifdef SWIGOCTAVE
    /* (Octave converts single element arrays to scalars and our typemaps take that for real) */
    %extend CDynamicObjectArray {
        bool append_element_real_vector(float64_t v, const char* name="")
        {
            SGVector<float64_t> wrap(1);
            wrap[0] = v;
            return $self->append_element(wrap, name);
        }
    }
#endif
#endif
#ifdef USE_FLOAT32
    %template(append_element_float_vector) CDynamicObjectArray::append_element_vector<SGVector<float32_t>, SGVector<float32_t>>;
    %template(append_element_float_matrix) CDynamicObjectArray::append_element_matrix<SGMatrix<float32_t>, SGMatrix<float32_t>>;
    %template(append_element_float) CDynamicObjectArray::append_element<float32_t, float32_t>;
#endif
#ifdef USE_INT32
    %template(append_element_int) CDynamicObjectArray::append_element<int32_t, int32_t>;
#endif
	%template(append_element_string_char_list) CDynamicObjectArray::append_element_string_list<SGStringList<char>, SGStringList<char>>;
	%template(append_element_string_word_list) CDynamicObjectArray::append_element_string_list<SGStringList<uint16_t>, SGStringList<uint16_t>>;
}
%include <shogun/lib/IndexBlock.h>
%include <shogun/lib/IndexBlockRelation.h>
%include <shogun/lib/IndexBlockGroup.h>
%include <shogun/lib/IndexBlockTree.h>
%include <shogun/lib/Data.h>
