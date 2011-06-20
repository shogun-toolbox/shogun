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

#include <shogun/lib/DataType.h>
%}

/* One dimensional input arrays */
%define TYPEMAP_IN_SGVECTOR(oct_type_check, oct_type, oct_converter, sg_type, if_type, error_string)
%typemap(typecheck, precedence=SWIG_TYPECHECK_POINTER)
        shogun::SGVector<sg_type>
{
    const octave_value m=$input;

    $1 = (m.is_matrix_type() && m.oct_type_check() && m.rows()==1) ? 1 : 0;
}

%typemap(in) shogun::SGVector<sg_type>
{
    oct_type m;
    const octave_value mat_feat=$input;
    if (!mat_feat.is_matrix_type() || !mat_feat.oct_type_check() || mat_feat.rows()!=1)
    {
        /*SG_ERROR("Expected " error_string " Vector as argument\n");*/
        SWIG_fail;
    }

    m = mat_feat.oct_converter();
    $1 = shogun::SGVector<sg_type>((sg_type*) m.fortran_vec(), m.cols());
}
%typemap(freearg) shogun::SGVector<sg_type>
{
}
%enddef

/* Define concrete examples of the TYPEMAP_SG_VECTOR macros */
TYPEMAP_IN_SGVECTOR(is_uint8_type, uint8NDArray, uint8_array_value, uint8_t, uint8_t, "Byte")
TYPEMAP_IN_SGVECTOR(is_char_matrix, charMatrix, char_matrix_value, char, char, "Char")
TYPEMAP_IN_SGVECTOR(is_int32_type, int32NDArray, int32_array_value, int32_t, int32_t, "Integer")
TYPEMAP_IN_SGVECTOR(is_int16_type, int16NDArray, int16_array_value, int16_t, int16_t, "Short")
TYPEMAP_IN_SGVECTOR(is_single_type, Matrix, matrix_value, float32_t, float32_t, "Single Precision")
TYPEMAP_IN_SGVECTOR(is_double_type, Matrix, matrix_value, float64_t, float64_t, "Double Precision")
TYPEMAP_IN_SGVECTOR(is_uint16_type, uint16NDArray, uint16_array_value, uint16_t, uint16_t, "Word")

#undef TYPEMAP_IN_SGVECTOR

%define TYPEMAP_OUT_SGVECTOR(oct_type, sg_type, if_type, error_string)
/* One dimensional output arrays */
%typemap(out) shogun::SGVector<sg_type>
{
    sg_type* vec = $1.vector;
    int32_t len = $1.vlen;

    oct_type mat=oct_type(dim_vector(1, len));

    if (mat.cols() != len)
        SWIG_fail;

    for (int32_t i=0; i<len; i++)
        mat(i) = (if_type) vec[i];

    $1.free_vector();

    $result=mat;
}
%enddef

/* Define concrete examples of the TYPEMAP_OUT_SGVECTOR macros */
TYPEMAP_OUT_SGVECTOR(uint8NDArray, uint8_t, uint8_t, "Byte")
TYPEMAP_OUT_SGVECTOR(charMatrix, char, char, "Char")
TYPEMAP_OUT_SGVECTOR(int32NDArray, int32_t, int32_t, "Integer")
TYPEMAP_OUT_SGVECTOR(int16NDArray, int16_t, int16_t, "Short")
TYPEMAP_OUT_SGVECTOR(Matrix, float32_t, float32_t, "Single Precision")
TYPEMAP_OUT_SGVECTOR(Matrix, float64_t, float64_t, "Double Precision")
TYPEMAP_OUT_SGVECTOR(uint16NDArray, uint16_t, uint16_t, "Word")

#undef TYPEMAP_OUT_SGVECTOR


/* Two dimensional input arrays */
%define TYPEMAP_IN_SGMATRIX(oct_type_check, oct_type, oct_converter, sg_type, if_type, error_string)
%typemap(typecheck, precedence=SWIG_TYPECHECK_POINTER)
        shogun::SGMatrix<sg_type>
{
    const octave_value m=$input;
    $1 = (m.is_matrix_type() && m.oct_type_check()) ? 1 : 0;
}

%typemap(in) shogun::SGMatrix<sg_type>
{
    oct_type m;
    const octave_value mat_feat=$input;
    if (!mat_feat.is_matrix_type() || !mat_feat.oct_type_check())
    {
        /*SG_ERROR("Expected " error_string " Matrix as argument\n");*/
        SWIG_fail;
    }

    m = mat_feat.oct_converter();

    $1 = shogun::SGMatrix<sg_type>((sg_type*) m.fortran_vec(), m.rows(), m.cols());
}
%typemap(freearg) shogun::SGMatrix<sg_type>
{
}
%enddef

/* Define concrete examples of the TYPEMAP_IN_SGMATRIX macros */
TYPEMAP_IN_SGMATRIX(is_uint8_type, uint8NDArray, uint8_array_value, uint8_t, uint8_t, "Byte")
TYPEMAP_IN_SGMATRIX(is_char_matrix, charMatrix, char_matrix_value, char, char, "Char")
TYPEMAP_IN_SGMATRIX(is_int32_type, int32NDArray, int32_array_value, int32_t, int32_t, "Integer")
TYPEMAP_IN_SGMATRIX(is_int16_type, int16NDArray, int16_array_value, int16_t, int16_t, "Short")
TYPEMAP_IN_SGMATRIX(is_single_type, Matrix, matrix_value, float32_t, float32_t, "Single Precision")
TYPEMAP_IN_SGMATRIX(is_double_type, Matrix, matrix_value, float64_t, float64_t, "Double Precision")
TYPEMAP_IN_SGMATRIX(is_uint16_type, uint16NDArray, uint16_array_value, uint16_t, uint16_t, "Word")

#undef TYPEMAP_IN_SGMATRIX

/* Two dimensional output arrays */
%define TYPEMAP_OUT_SGMATRIX(oct_type, sg_type, if_type, error_string)
%typemap(out) shogun::SGMatrix<sg_type>
{
    sg_type* matrix = $1.matrix;
    int32_t num_feat = $1.num_rows;
    int32_t num_vec = $1.num_cols;

    oct_type mat=oct_type(dim_vector(num_feat, num_vec));

    if (mat.rows() != num_feat || mat.cols() != num_vec)
        SWIG_fail;

    for (int32_t i=0; i<num_vec; i++)
    {
        for (int32_t j=0; j<num_feat; j++)
            mat(j,i) = (if_type) matrix[j+i*num_feat];
    }

    $1.free_matrix();

    $result=mat;
}
%enddef

TYPEMAP_OUT_SGMATRIX(uint8NDArray, uint8_t, uint8_t, "Byte")
TYPEMAP_OUT_SGMATRIX(charMatrix, char, char, "Char")
TYPEMAP_OUT_SGMATRIX(int32NDArray, int32_t, int32_t, "Integer")
TYPEMAP_OUT_SGMATRIX(int16NDArray, int16_t, int16_t, "Short")
TYPEMAP_OUT_SGMATRIX(Matrix, float32_t, float32_t, "Single Precision")
TYPEMAP_OUT_SGMATRIX(Matrix, float64_t, float64_t, "Double Precision")
TYPEMAP_OUT_SGMATRIX(uint16NDArray, uint16_t, uint16_t, "Word")
#undef TYPEMAP_OUT_SGMATRIX


/* TODO INND ARRAYS */

/* input typemap for CStringFeatures<char> etc */
%define TYPEMAP_STRINGFEATURES_IN(oct_type_check, oct_type, oct_converter, sg_type, if_type, error_string)
%typemap(in) shogun::SGStringList<sg_type>
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
    SGStringList<sg_type> sl;
    sl.strings=strings;
    sl.num_strings=num_strings;
    sl.max_string_length=max_len;
    $1 = sl;
}
%enddef

TYPEMAP_STRINGFEATURES_IN(is_matrix_type() && arg.is_uint8_type, uint8NDArray, uint8_array_value, uint8_t, uint8_t, "Byte")
TYPEMAP_STRINGFEATURES_IN(is_char_matrix, charMatrix, char_matrix_value, char, char, "Char")
TYPEMAP_STRINGFEATURES_IN(is_matrix_type() && arg.is_int32_type, int32NDArray, int32_array_value, int32_t, int32_t, "Integer")
TYPEMAP_STRINGFEATURES_IN(is_matrix_type() && arg.is_int16_type, int16NDArray, int16_array_value, int16_t, int16_t, "Short")
TYPEMAP_STRINGFEATURES_IN(is_matrix_type() && arg.is_uint16_type, uint16NDArray, uint16_array_value, uint16_t, uint16_t, "Word")
#undef TYPEMAP_STRINGFEATURES_IN

/* output typemap for CStringFeatures */
%define TYPEMAP_STRINGFEATURES_OUT(type,typecode)
%typemap(out) shogun::SGStringList<type>
{
    /* TODO STRING OUT TYPEMAPS */
}
%enddef

TYPEMAP_STRINGFEATURES_OUT(char,          charMatrix)
TYPEMAP_STRINGFEATURES_OUT(uint8_t,       uint8NDArray)
TYPEMAP_STRINGFEATURES_OUT(int16_t,       int16NDArray)
TYPEMAP_STRINGFEATURES_OUT(uint16_t,      uint16NDArray)
TYPEMAP_STRINGFEATURES_OUT(int32_t,       int32NDArray)
TYPEMAP_STRINGFEATURES_OUT(uint32_t,      uint32NDArray)
TYPEMAP_STRINGFEATURES_OUT(int64_t,       int64NDArray)
TYPEMAP_STRINGFEATURES_OUT(uint64_t,      uint64NDArray)
TYPEMAP_STRINGFEATURES_OUT(float64_t,     Matrix)

#undef TYPEMAP_STRINGFEATURES_OUT

/* input typemap for Sparse Features */
%define TYPEMAP_SPARSEFEATURES_IN(type,typecode)
%typemap(in) shogun::SGSparseMatrix<type>
{
    /* TODO SPARSE MATRIX IN */
}
%enddef
TYPEMAP_SPARSEFEATURES_IN(float64_t,     Matrix)
#undef TYPEMAP_SPARSEFEATURES_IN

/* output typemap for sparse features returns (data, row, ptr) */
%define TYPEMAP_SPARSEFEATURES_OUT(type,typecode)
%typemap(out) shogun::SGSparseMatrix<type>
{
    /* TODO SPARSE MATRIX OUT */
}
%enddef

TYPEMAP_SPARSEFEATURES_OUT(float64_t,     NPY_FLOAT64)
#undef TYPEMAP_SPARSEFEATURES_OUT
