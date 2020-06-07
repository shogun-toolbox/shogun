/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Viktor Gal, Shubham Shukla, Sergey Lisitsyn
 */

#ifndef SWIGPYTHON
#define PROTOCOLS_SGVECTOR(class_name, type_name, format_str, typecode)
#endif

%shared_ptr(shogun::Signal)
%shared_ptr(shogun::Time)
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

%ignore RADIX_STACK_SIZE;
%ignore NUMTRAPPEDSIGS;
%ignore TRIE_TERMINAL_CHARACTER;
%ignore NO_CHILD;

/* Include Class Headers to make them visible from within the target language */
%include <shogun/lib/DataType.h>
%include <shogun/lib/SGReferencedData.h>
%include <shogun/lib/SGVector.h>
%include <shogun/lib/SGMatrix.h>
%include <shogun/lib/SGSparseVector.h>
%include <shogun/lib/SGSparseMatrix.h>
%include <shogun/lib/Tokenizer.h>
%include <shogun/lib/DelimiterTokenizer.h>
%include <shogun/lib/NGramTokenizer.h>
%include <shogun/lib/Signal.h>
%include <shogun/lib/Time.h>
%include <shogun/lib/Compressor.h>
%include <shogun/lib/StructuredDataTypes.h>
%include <shogun/lib/StructuredData.h>
%include <shogun/lib/DynamicObjectArray.h>

%define SGCONTAINER_NOVALUE_WRAPPER(TYPE)
%feature("novaluewrapper") shogun::SGVector<TYPE>;
%feature("novaluewrapper") shogun::SGMatrix<TYPE>;
%feature("novaluewrapper") shogun::SGSparseVector<TYPE>;
%feature("novaluewrapper") shogun::SGSparseMatrix<TYPE>;
%template() shogun::SGVector<TYPE>;
%template() shogun::SGMatrix<TYPE>;
%template() shogun::SGSparseVector<TYPE>;
%template() shogun::SGSparseMatrix<TYPE>;
%enddef

SGCONTAINER_NOVALUE_WRAPPER(bool)
SGCONTAINER_NOVALUE_WRAPPER(char)
SGCONTAINER_NOVALUE_WRAPPER(uint8_t)
SGCONTAINER_NOVALUE_WRAPPER(uint16_t)
SGCONTAINER_NOVALUE_WRAPPER(uint32_t)
SGCONTAINER_NOVALUE_WRAPPER(uint64_t)
SGCONTAINER_NOVALUE_WRAPPER(int16_t)
SGCONTAINER_NOVALUE_WRAPPER(int32_t)
SGCONTAINER_NOVALUE_WRAPPER(int64_t)
SGCONTAINER_NOVALUE_WRAPPER(float32_t)
SGCONTAINER_NOVALUE_WRAPPER(float64_t)
SGCONTAINER_NOVALUE_WRAPPER(floatmax_t)
SGCONTAINER_NOVALUE_WRAPPER(std::complex<double>)

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
