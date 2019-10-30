/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Viktor Gal, Shubham Shukla, Sergey Lisitsyn
 */

#ifndef SWIGPYTHON
#define PROTOCOLS_SGVECTOR(class_name, type_name, format_str, typecode)
#endif

%shared_ptr(shogun::Cache)
%shared_ptr(shogun::ListElement)
%shared_ptr(shogun::Signal)
%shared_ptr(shogun::Time)
%shared_ptr(shogun::Hash)
%shared_ptr(shogun::Compressor)
%shared_ptr(shogun::StructuredData)
%shared_ptr(shogun::DynamicObjectArray)
%shared_ptr(shogun::Tokenizer)
%shared_ptr(shogun::DelimiterTokenizer)
%shared_ptr(shogun::NGramTokenizer)

%shared_ptr(shogun::IndexBlock)
%shared_ptr(shogun::IndexBlockRelation)
%shared_ptr(shogun::IndexBlockGroup)
%shared_ptr(shogun::IndexBlockTree)
%shared_ptr(shogun::Data)

#ifdef USE_CHAR
%shared_ptr(shogun::DynamicArray<char>)
#endif
#ifdef USE_UINT8
%shared_ptr(shogun::DynamicArray<uint8_t>)
#endif
#ifdef USE_INT16
%shared_ptr(shogun::DynamicArray<int16_t>)
#endif
#ifdef USE_UINT16
%shared_ptr(shogun::DynamicArray<uint16_t>)
#endif
#ifdef USE_INT32
%shared_ptr(shogun::DynamicArray<int32_t>)
#endif
#ifdef USE_UINT32
%shared_ptr(shogun::DynamicArray<uint32_t>)
#endif
#ifdef USE_INT64
%shared_ptr(shogun::DynamicArray<int64_t>)
#endif
#ifdef USE_UINT64
%shared_ptr(shogun::DynamicArray<uint64_t>)
#endif
#ifdef USE_FLOAT32
%shared_ptr(shogun::DynamicArray<float32_t>)
#endif
#ifdef USE_FLOAT64
%shared_ptr(shogun::DynamicArray<float64_t>)
#endif

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
namespace shogun
{
#ifdef USE_CHAR
        %template(DynamicCharArray) DynamicArray<char>;
#endif
#ifdef USE_UINT8
        %template(DynamicByteArray) DynamicArray<uint8_t>;
#endif
#ifdef USE_INT16
        %template(DynamicShortArray) DynamicArray<int16_t>;
#endif
#ifdef USE_UINT16
        %template(DynamicWordArray) DynamicArray<uint16_t>;
#endif
#ifdef USE_INT32
        %template(DynamicIntArray) DynamicArray<int32_t>;
#endif
#ifdef USE_UINT32
        %template(DynamicUIntArray) DynamicArray<uint32_t>;
#endif
#ifdef USE_INT64
        %template(DynamicLongArray) DynamicArray<int64_t>;
#endif
#ifdef USE_UINT64
        %template(DynamicULongArray) DynamicArray<uint64_t>;
#endif
#ifdef USE_FLOAT32
        %template(DynamicShortRealArray) DynamicArray<float32_t>;
#endif
#ifdef USE_FLOAT64
        %template(DynamicRealArray) DynamicArray<float64_t>;
#endif
}
/* Hash */
%include <shogun/lib/Hash.h>

%include <shogun/lib/Tokenizer.h>
%include <shogun/lib/DelimiterTokenizer.h>
%include <shogun/lib/NGramTokenizer.h>
%include <shogun/lib/Cache.h>
%include <shogun/lib/Signal.h>
%include <shogun/lib/Time.h>
%include <shogun/lib/Trie.h>
%include <shogun/lib/Compressor.h>
%include <shogun/lib/StructuredDataTypes.h>
%include <shogun/lib/StructuredData.h>
%include <shogun/lib/DynamicObjectArray.h>
namespace shogun
{
%extend DynamicObjectArray
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

    template <typename T>
    void append_element_string_list(const std::vector<SGVector<T>>& v, const char* name="")
    {
        $self->append_element(v, name);
    }
}

    /* Specialize DynamicObjectArray::append_element function */
#ifdef USE_FLOAT64
    %template(append_element_real_vector) DynamicObjectArray::append_element_vector<SGVector<float64_t>, SGVector<float64_t>>;
    %template(append_element_real_matrix) DynamicObjectArray::append_element_matrix<SGMatrix<float64_t>, SGMatrix<float64_t>>;
    %template(append_element_real) DynamicObjectArray::append_element<float64_t, float64_t>;
#ifdef SWIGOCTAVE
    /* (Octave converts single element arrays to scalars and our typemaps take that for real) */
    %extend DynamicObjectArray {
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
    %template(append_element_float_vector) DynamicObjectArray::append_element_vector<SGVector<float32_t>, SGVector<float32_t>>;
    %template(append_element_float_matrix) DynamicObjectArray::append_element_matrix<SGMatrix<float32_t>, SGMatrix<float32_t>>;
    %template(append_element_float) DynamicObjectArray::append_element<float32_t, float32_t>;
#endif
#ifdef USE_INT32
    %template(append_element_int) DynamicObjectArray::append_element<int32_t, int32_t>;
#endif
    %template(append_element_string_char_list) DynamicObjectArray::append_element_string_list<char>;
    %template(append_element_string_word_list) DynamicObjectArray::append_element_string_list<uint16_t>;
}
%include <shogun/lib/IndexBlock.h>
%include <shogun/lib/IndexBlockRelation.h>
%include <shogun/lib/IndexBlockGroup.h>
%include <shogun/lib/IndexBlockTree.h>
%include <shogun/lib/Data.h>
