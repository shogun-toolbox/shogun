/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * This code is inspired by the python numpy.i typemaps, from John Hunter
 * and Bill Spotz.
 *
 * Written (W) 2008-2009 Soeren Sonnenburg
 * Copyright (C) 2008-2009 Fraunhofer Institute FIRST and Max-Planck-Society
 */

%{
#include <octave/config.h>

#include <octave/ov.h>
#include <octave/defun-dld.h>
#include <octave/error.h>
#include <octave/oct-obj.h>
#include <octave/pager.h>
#include <octave/symtab.h>
#include <octave/variables.h>
#include <octave/Cell.h>
%}

/* TYPEMAP_IN macros
 *
 * This family of typemaps allows pure input C arguments of the form
 *
 *     (type* IN_ARRAY1, int32_t DIM1)
 *     (type* IN_ARRAY2, int32_t DIM1, int32_t DIM2)
 *
 * where "type" is any type supported by the numpy module, to be
 * called in python with an argument list of a single array (or any
 * python object that can be passed to the numpy.array constructor
 * to produce an arrayof te specified shape).  This can be applied to
 * a existing functions using the %apply directive:
 *
 *     %apply (float64_t* IN_ARRAY1, int32_t DIM1) {float64_t* series, int32_t length}
 *     %apply (float64_t* IN_ARRAY2, int32_t DIM1, int32_t DIM2) {float64_t* mx, int32_t rows, int32_t cols}
 *     float64_t sum(float64_t* series, int32_t length);
 *     float64_t max(float64_t* mx, int32_t rows, int32_t cols);
 *
 * or with
 *
 *     float64_t sum(float64_t* IN_ARRAY1, int32_t DIM1);
 *     float64_t max(float64_t* IN_ARRAY2, int32_t DIM1, int32_t DIM2);
 */

/* One dimensional input arrays */
%define TYPEMAP_IN1(oct_type_check, oct_type, oct_converter, sg_type, if_type, error_string)
%typemap(typecheck, precedence=SWIG_TYPECHECK_POINTER)
        (sg_type* IN_ARRAY1, int32_t DIM1)
{
    const octave_value m=$input;

    $1 = (m.is_matrix_type() && m.oct_type_check() && m.rows()==1) ? 1 : 0;
}

%typemap(in) (sg_type* IN_ARRAY1, int32_t DIM1) (oct_type m)
{
    const octave_value mat_feat=$input;
    if (!mat_feat.is_matrix_type() || !mat_feat.oct_type_check() || mat_feat.rows()!=1)
    {
        /*SG_ERROR("Expected " error_string " Vector as argument\n");*/
        SWIG_fail;
    }

    m = mat_feat.oct_converter();
    $1 = (sg_type*) m.fortran_vec();
    $2 = m.cols();
}
%typemap(freearg) (type* IN_ARRAY1, int32_t DIM1) {
}
%enddef

/* Define concrete examples of the TYPEMAP_IN1 macros */
TYPEMAP_IN1(is_uint8_type, uint8NDArray, uint8_array_value, uint8_t, uint8_t, "Byte")
TYPEMAP_IN1(is_char_matrix, charMatrix, char_matrix_value, char, char, "Char")
TYPEMAP_IN1(is_int32_type, int32NDArray, int32_array_value, int32_t, int32_t, "Integer")
TYPEMAP_IN1(is_int16_type, int16NDArray, int16_array_value, int16_t, int16_t, "Short")
TYPEMAP_IN1(is_single_type, Matrix, matrix_value, float32_t, float32_t, "Single Precision")
TYPEMAP_IN1(is_double_type, Matrix, matrix_value, float64_t, float64_t, "Double Precision")
TYPEMAP_IN1(is_uint16_type, uint16NDArray, uint16_array_value, uint16_t, uint16_t, "Word")
#undef TYPEMAP_IN1


%define TYPEMAP_IN2(oct_type_check, oct_type, oct_converter, sg_type, if_type, error_string)
%typemap(typecheck, precedence=SWIG_TYPECHECK_POINTER)
        (sg_type* IN_ARRAY2, int32_t DIM1, int32_t DIM2)
{
    const octave_value m=$input;
    $1 = (m.is_matrix_type() && m.oct_type_check()) ? 1 : 0;
}

%typemap(in) (sg_type* IN_ARRAY2, int32_t DIM1, int32_t DIM2) (oct_type m)
{
    const octave_value mat_feat=$input;
    if (!mat_feat.is_matrix_type() || !mat_feat.oct_type_check())
    {
        /*SG_ERROR("Expected " error_string " Matrix as argument\n");*/
        SWIG_fail;
    }

    m = mat_feat.oct_converter();

    $1 = (sg_type*) m.fortran_vec();
    $2 = m.rows();
    $3 = m.cols();
}
%typemap(freearg) (type* IN_ARRAY2, int32_t DIM1, int32_t DIM2) {
}
%enddef

TYPEMAP_IN2(is_uint8_type, uint8NDArray, uint8_array_value, uint8_t, uint8_t, "Byte")
TYPEMAP_IN2(is_char_matrix, charMatrix, char_matrix_value, char, char, "Char")
TYPEMAP_IN2(is_int32_type, int32NDArray, int32_array_value, int32_t, int32_t, "Integer")
TYPEMAP_IN2(is_int16_type, int16NDArray, int16_array_value, int16_t, int16_t, "Short")
TYPEMAP_IN2(is_single_type, Matrix, matrix_value, float32_t, float32_t, "Single Precision")
TYPEMAP_IN2(is_double_type, Matrix, matrix_value, float64_t, float64_t, "Double Precision")
TYPEMAP_IN2(is_uint16_type, uint16NDArray, uint16_array_value, uint16_t, uint16_t, "Word")
#undef TYPEMAP_IN2

/* TYPEMAP_ARGOUT macros
 *
 * This family of typemaps allows output C arguments of the form
 *
 *     (type** ARGOUT_ARRAY)
 *
 * where "type" is any type supported by the numpy module, to be
 * called in python with an argument list of a single contiguous
 * numpy array.  This can be applied to an existing function using
 * the %apply directive:
 *
 *     %apply (float64_t** ARGOUT_ARRAY1, {(float64_t** series, int32_t* len)}
 *     %apply (float64_t** ARGOUT_ARRAY2, {(float64_t** matrix, int32_t* d1, int32_t* d2)}
 *
 * with
 *
 *     void sum(float64_t* series, int32_t* len);
 *     void sum(float64_t** series, int32_t* len);
 *     void sum(float64_t** matrix, int32_t* d1, int32_t* d2);
 *
 * where sum mallocs the array and assigns dimensions and the pointer
 *
 */
%define TYPEMAP_ARGOUT1(oct_type, sg_type, if_type, error_string)
%typemap(in, numinputs=0) (sg_type** ARGOUT1, int32_t* DIM1) {
    $1 = (sg_type**) malloc(sizeof(sg_type*));
    $2 = (int32_t*) malloc(sizeof(int32_t));
}

%typemap(argout) (sg_type** ARGOUT1, int32_t* DIM1) {
    sg_type* vec = *$1;
    int32_t len = *$2;

    oct_type mat=oct_type(dim_vector(1, len));

    if (mat.cols() != len)
        SWIG_fail;

    for (int32_t i=0; i<len; i++)
        mat(i) = (if_type) vec[i];

    $result->append(mat);
    free(*$1); free($1); free($2);
}
%enddef

TYPEMAP_ARGOUT1(uint8NDArray, uint8_t, uint8_t, "Byte")
TYPEMAP_ARGOUT1(charMatrix, char, char, "Char")
TYPEMAP_ARGOUT1(int32NDArray, int32_t, int32_t, "Integer")
TYPEMAP_ARGOUT1(int16NDArray, int16_t, int16_t, "Short")
TYPEMAP_ARGOUT1(Matrix, float32_t, float32_t, "Single Precision")
TYPEMAP_ARGOUT1(Matrix, float64_t, float64_t, "Double Precision")
TYPEMAP_ARGOUT1(uint16NDArray, uint16_t, uint16_t, "Word")

#undef TYPEMAP_ARGOUT1

%define TYPEMAP_ARGOUT2(oct_type, sg_type, if_type, error_string)
%typemap(in, numinputs=0) (sg_type** ARGOUT2, int32_t* DIM1, int32_t* DIM2) {
    $1 = (sg_type**) malloc(sizeof(sg_type*));
    $2 = (int32_t*) malloc(sizeof(int32_t));
    $3 = (int32_t*) malloc(sizeof(int32_t));
}

%typemap(argout) (sg_type** ARGOUT2, int32_t* DIM1, int32_t* DIM2) {
    sg_type* matrix = *$1;
    int32_t num_feat = *$2;
    int32_t num_vec = *$3;

    oct_type mat=oct_type(dim_vector(num_feat, num_vec));

    if (mat.rows() != num_feat || mat.cols() != num_vec)
        SWIG_fail;

    for (int32_t i=0; i<num_vec; i++)
    {
        for (int32_t j=0; j<num_feat; j++)
            mat(j,i) = (if_type) matrix[j+i*num_feat];
    }

    free(*$1); free($1); free($2); free($3);
    $result->append(mat);
}
%enddef

TYPEMAP_ARGOUT2(uint8NDArray, uint8_t, uint8_t, "Byte")
TYPEMAP_ARGOUT2(charMatrix, char, char, "Char")
TYPEMAP_ARGOUT2(int32NDArray, int32_t, int32_t, "Integer")
TYPEMAP_ARGOUT2(int16NDArray, int16_t, int16_t, "Short")
TYPEMAP_ARGOUT2(Matrix, float32_t, float32_t, "Single Precision")
TYPEMAP_ARGOUT2(Matrix, float64_t, float64_t, "Double Precision")
TYPEMAP_ARGOUT2(uint16NDArray, uint16_t, uint16_t, "Word")
#undef TYPEMAP_ARGOUT2

/* input typemap for CStringFeatures<char> etc */
%define TYPEMAP_STRINGFEATURES_IN(oct_type_check, oct_type, oct_converter, sg_type, if_type, error_string)
%typemap(in) (shogun::SGString<sg_type>* IN_STRINGS, int32_t NUM, int32_t MAXLEN)
{
    using namespace shogun;
    int32_t max_len=0;
    int32_t num_strings=0;
    SGString<sg_type>* strings=NULL;

    octave_value arg=$input;
    if (arg.is_cell())
    {
        Cell c = arg.cell_value();
        num_strings=c.nelem();
        ASSERT(num_strings>=1);
        strings=new SGString<sg_type>[num_strings];

        for (int32_t i=0; i<num_strings; i++)
        {
            if (!c.elem(i).oct_type_check() || !c.elem(i).rows()==1)
            {
                /* SG_ERROR("Expected String of type " error_string " as argument %d.\n", m_rhs_counter);*/
                SWIG_fail;
            }
            oct_type str=c.elem(i).oct_converter();

            int32_t len=str.cols();
            if (len>0) 
            { 
                strings[i].length=len; /* all must have same length in octave */
                strings[i].string=new sg_type[len+1]; /* not zero terminated in octave */

                int32_t j; 
                for (j=0; j<len; j++)
                    strings[i].string[j]=str(0,j);
                strings[i].string[j]='\0';
                max_len=CMath::max(max_len, len);
            }
            else
            {
                /*SG_WARNING( "string with index %d has zero length.\n", i+1);*/
                strings[i].length=0;
                strings[i].string=NULL;
            }
        }
    }
    else if (arg.oct_type_check())
    {
        oct_type data=arg.oct_converter();
        num_strings=data.cols(); 
        int32_t len=data.rows();
        strings=new SGString<sg_type>[num_strings];
        ASSERT(strings);

        for (int32_t i=0; i<num_strings; i++)
        { 
            if (len>0) 
            { 
                strings[i].length=len; /* all must have same length in octave */
                strings[i].string=new sg_type[len+1]; /* not zero terminated in octave */

                int32_t j;
                for (j=0; j<len; j++)
                    strings[i].string[j]=data(j,i);
                strings[i].string[j]='\0';
            }
            else
            { 
                /*SG_WARNING( "string with index %d has zero length.\n", i+1);*/
                strings[i].length=0;
                strings[i].string=NULL;
            }
        }
        max_len=len;
    }
    else
    {
        /*SG_PRINT("matrix_type: %d\n", arg.is_matrix_type() ? 1 : 0);
        SG_ERROR("Expected String, got class %s as argument.\n",
                "???");*/
        SWIG_fail;
    }
    $1 = strings;
    $2 = num_strings;
    $3 = max_len;
}
%enddef

TYPEMAP_STRINGFEATURES_IN(is_matrix_type() && arg.is_uint8_type, uint8NDArray, uint8_array_value, uint8_t, uint8_t, "Byte")
TYPEMAP_STRINGFEATURES_IN(is_char_matrix, charMatrix, char_matrix_value, char, char, "Char")
TYPEMAP_STRINGFEATURES_IN(is_matrix_type() && arg.is_int32_type, int32NDArray, int32_array_value, int32_t, int32_t, "Integer")
TYPEMAP_STRINGFEATURES_IN(is_matrix_type() && arg.is_int16_type, int16NDArray, int16_array_value, int16_t, int16_t, "Short")
TYPEMAP_STRINGFEATURES_IN(is_matrix_type() && arg.is_uint16_type, uint16NDArray, uint16_array_value, uint16_t, uint16_t, "Word")
#undef TYPEMAP_STRINGFEATURES_IN

/* output typemap for CStringFeatures */
%define TYPEMAP_STRINGFEATURES_ARGOUT(type,typecode)
%typemap(in, numinputs=0) (shogun::SGString<type>** ARGOUT_STRINGS, int32_t* NUM) {
    $1 = (shogun::SGString<type>**) malloc(sizeof(shogun::SGString<type>*));
    $2 = (int32_t*) malloc(sizeof(int32_t));
}
%typemap(argout) (shogun::SGString<type>** ARGOUT_STRINGS, int32_t* NUM) {
    if (!$1 || !$2)
        SWIG_fail;
    free($1); free($2);
}
%enddef

TYPEMAP_STRINGFEATURES_ARGOUT(char,          charMatrix)
TYPEMAP_STRINGFEATURES_ARGOUT(uint8_t,       uint8NDArray)
TYPEMAP_STRINGFEATURES_ARGOUT(int16_t,       int16NDArray)
TYPEMAP_STRINGFEATURES_ARGOUT(uint16_t,      uint16NDArray)
TYPEMAP_STRINGFEATURES_ARGOUT(int32_t,       int32NDArray)
TYPEMAP_STRINGFEATURES_ARGOUT(uint32_t,      uint32NDArray)
TYPEMAP_STRINGFEATURES_ARGOUT(int64_t,       int64NDArray)
TYPEMAP_STRINGFEATURES_ARGOUT(uint64_t,      uint64NDArray)
TYPEMAP_STRINGFEATURES_ARGOUT(float64_t,     Matrix)
#undef TYPEMAP_STRINGFEATURES_ARGOUT

/* input typemap for Sparse Features */
%define TYPEMAP_SPARSEFEATURES_IN(type,typecode)
%typemap(in) (shogun::SGSparseVector<type>* IN_SPARSE, int32_t DIM1, int32_t DIM2)
{
}
%enddef
TYPEMAP_SPARSEFEATURES_IN(float64_t,     Matrix)
#undef TYPEMAP_SPARSEFEATURES_IN

/* output typemap for sparse features returns (data, row, ptr) */
%define TYPEMAP_SPARSEFEATURES_ARGOUT(type,typecode)
%typemap(in, numinputs=0) (shogun::SGSparseVector<type>** ARGOUT_SPARSE, int32_t* DIM1, int32_t* DIM2, int64_t* NNZ) {
    using namespace shogun;

    $1 = (SGSparseVector<type>**) malloc(sizeof(SGSparseVector<type>*));
    $2 = (int32_t*) malloc(sizeof(int32_t));
    $3 = (int32_t*) malloc(sizeof(int32_t));
    $4 = (int64_t*) malloc(sizeof(int64_t));
}
%typemap(argout) (shogun::SGSparseVector<type>** ARGOUT_SPARSE, int32_t* DIM1, int32_t* DIM2, int64_t* NNZ) {
    if (!$1 || !$2 || !$3 || !$4)
        SWIG_fail;
    free($1); free($2); free($3); free($4);
}
%enddef

TYPEMAP_SPARSEFEATURES_ARGOUT(float64_t,     NPY_FLOAT64)
#undef TYPEMAP_SPARSEFEATURES_ARGOUT
