/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * This code is inspired by the python numpy.i typemaps, from John Hunter
 * and Bill Spotz.
 *
 * Written (W) 2008 Soeren Sonnenburg
 * Copyright (C) 2008 Fraunhofer Institute FIRST and Max-Planck-Society
 */

%{
#include "lib/common.h"
#include "lib/octave.h"
%}

/* TYPEMAP_IN macros
 *
 * This family of typemaps allows pure input C arguments of the form
 *
 *     (type* IN_ARRAY1, int DIM1)
 *     (type* IN_ARRAY2, int DIM1, int DIM2)
 *
 * where "type" is any type supported by the numpy module, to be
 * called in python with an argument list of a single array (or any
 * python object that can be passed to the numpy.array constructor
 * to produce an arrayof te specified shape).  This can be applied to
 * a existing functions using the %apply directive:
 *
 *     %apply (double* IN_ARRAY1, int DIM1) {double* series, int length}
 *     %apply (double* IN_ARRAY2, int DIM1, int DIM2) {double* mx, int rows, int cols}
 *     double sum(double* series, int length);
 *     double max(double* mx, int rows, int cols);
 *
 * or with
 *
 *     double sum(double* IN_ARRAY1, int DIM1);
 *     double max(double* IN_ARRAY2, int DIM1, int DIM2);
 */

/* One dimensional input arrays */
%define TYPEMAP_IN1(oct_type_check, oct_type, oct_converter, sg_type, if_type, error_string)
%typemap(typecheck, precedence=SWIG_TYPECHECK_POINTER)
        (sg_type* IN_ARRAY1, INT DIM1)
{
    const octave_value m=$input;

    $1 = (m.is_matrix_type() && m.oct_type_check() && m.rows()==1) ? 1 : 0;
}

%typemap(in) (sg_type* IN_ARRAY1, INT DIM1) (oct_type m)
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
%typemap(freearg) (type* IN_ARRAY1, INT DIM1) {
}
%enddef

/* Define concrete examples of the TYPEMAP_IN1 macros */
TYPEMAP_IN1(is_uint8_type, uint8NDArray, uint8_array_value, BYTE, BYTE, "Byte")
TYPEMAP_IN1(is_char_matrix, charMatrix, char_matrix_value, char, char, "Char")
TYPEMAP_IN1(is_int32_type, int32NDArray, uint8_array_value, INT, INT, "Integer")
TYPEMAP_IN1(is_int16_type, int16NDArray, uint8_array_value, SHORT, SHORT, "Short")
TYPEMAP_IN1(is_single_type, Matrix, matrix_value, SHORTREAL, SHORTREAL, "Single Precision")
TYPEMAP_IN1(is_double_type, Matrix, matrix_value, DREAL, DREAL, "Double Precision")
TYPEMAP_IN1(is_uint16_type, uint16NDArray, uint16_array_value, WORD, WORD, "Word")
#undef TYPEMAP_IN1


%define TYPEMAP_IN2(oct_type_check, oct_type, oct_converter, sg_type, if_type, error_string)
%typemap(typecheck, precedence=SWIG_TYPECHECK_POINTER)
        (sg_type* IN_ARRAY2, INT DIM1, INT DIM2)
{
    const octave_value m=$input;
    $1 = (m.is_matrix_type() && m.oct_type_check()) ? 1 : 0;
}

%typemap(in) (sg_type* IN_ARRAY2, INT DIM1, INT DIM2) (oct_type m)
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
%typemap(freearg) (type* IN_ARRAY2, INT DIM1, INT DIM2) {
}
%enddef

TYPEMAP_IN2(is_uint8_type, uint8NDArray, uint8_array_value, BYTE, BYTE, "Byte")
TYPEMAP_IN2(is_char_matrix, charMatrix, char_matrix_value, char, char, "Char")
TYPEMAP_IN2(is_int32_type, int32NDArray, uint8_array_value, INT, INT, "Integer")
TYPEMAP_IN2(is_int16_type, int16NDArray, uint8_array_value, SHORT, SHORT, "Short")
TYPEMAP_IN2(is_single_type, Matrix, matrix_value, SHORTREAL, SHORTREAL, "Single Precision")
TYPEMAP_IN2(is_double_type, Matrix, matrix_value, DREAL, DREAL, "Double Precision")
TYPEMAP_IN2(is_uint16_type, uint16NDArray, uint16_array_value, WORD, WORD, "Word")
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
 *     %apply (DREAL** ARGOUT_ARRAY1, {(DREAL** series, INT* len)}
 *     %apply (DREAL** ARGOUT_ARRAY2, {(DREAL** matrix, INT* d1, INT* d2)}
 *
 * with
 *
 *     void sum(DREAL* series, INT* len);
 *     void sum(DREAL** series, INT* len);
 *     void sum(DREAL** matrix, INT* d1, INT* d2);
 *
 * where sum mallocs the array and assigns dimensions and the pointer
 *
 */
%define TYPEMAP_ARGOUT1(oct_type, sg_type, if_type, error_string)
%typemap(in, numinputs=0) (sg_type** ARGOUT1, INT* DIM1) {
    $1 = (sg_type**) malloc(sizeof(sg_type*));
    $2 = (INT*) malloc(sizeof(INT));
}

%typemap(argout) (sg_type** ARGOUT1, INT* DIM1) {
    sg_type* vec = *$1;
    INT len = *$2;

    oct_type mat=oct_type(dim_vector(1, len));

    if (mat.cols() != len)
        SWIG_fail;

    for (INT i=0; i<len; i++)
        mat(i) = (if_type) vec[i];

    $result->append(mat);
    free(*$1); free($1); free($2);
}
%enddef

TYPEMAP_ARGOUT1(uint8NDArray, BYTE, BYTE, "Byte")
TYPEMAP_ARGOUT1(charMatrix, char, char, "Char")
TYPEMAP_ARGOUT1(int32NDArray, INT, INT, "Integer")
TYPEMAP_ARGOUT1(int16NDArray, SHORT, SHORT, "Short")
TYPEMAP_ARGOUT1(Matrix, SHORTREAL, SHORTREAL, "Single Precision")
TYPEMAP_ARGOUT1(Matrix, DREAL, DREAL, "Double Precision")
TYPEMAP_ARGOUT1(uint16NDArray, WORD, WORD, "Word")

#undef TYPEMAP_ARGOUT1

%define TYPEMAP_ARGOUT2(oct_type, sg_type, if_type, error_string)
%typemap(in, numinputs=0) (sg_type** ARGOUT2, INT* DIM1, INT* DIM2) {
    $1 = (sg_type**) malloc(sizeof(sg_type*));
    $2 = (INT*) malloc(sizeof(INT));
    $3 = (INT*) malloc(sizeof(INT));
}

%typemap(argout) (sg_type** ARGOUT2, INT* DIM1, INT* DIM2) {
    sg_type* matrix = *$1;
    INT num_feat = *$2;
    INT num_vec = *$3;

    oct_type mat=oct_type(dim_vector(num_feat, num_vec));

    if (mat.rows() != num_feat || mat.cols() != num_vec)
        SWIG_fail;

    for (INT i=0; i<num_vec; i++)
    {
        for (INT j=0; j<num_feat; j++)
            mat(j,i) = (if_type) matrix[j+i*num_feat];
    }

    free(*$1); free($1); free($2); free($3);
    $result->append(mat);
}
%enddef

TYPEMAP_ARGOUT2(uint8NDArray, BYTE, BYTE, "Byte")
TYPEMAP_ARGOUT2(charMatrix, char, char, "Char")
TYPEMAP_ARGOUT2(int32NDArray, INT, INT, "Integer")
TYPEMAP_ARGOUT2(int16NDArray, SHORT, SHORT, "Short")
TYPEMAP_ARGOUT2(Matrix, SHORTREAL, SHORTREAL, "Single Precision")
TYPEMAP_ARGOUT2(Matrix, DREAL, DREAL, "Double Precision")
TYPEMAP_ARGOUT2(uint16NDArray, WORD, WORD, "Word")
#undef TYPEMAP_ARGOUT2

/* input typemap for CStringFeatures<char> etc */
%define GET_STRINGLIST(oct_type_check, oct_type, oct_converter, sg_type, if_type, error_string)
%typemap(in) (T_STRING<sg_type>* strings, INT num_strings, INT max_len)
{
    INT max_len=0;
    INT num_strings=0;
    T_STRING<sg_type>* strings=NULL;

    octave_value arg=$input;
    if (arg.is_cell())
    {
        Cell c = arg.cell_value();
        num_strings=c.nelem();
        ASSERT(num_strings>=1);
        strings=new T_STRING<sg_type>[num_strings];

        for (int i=0; i<num_strings; i++)
        {
            if (!c.elem(i).oct_type_check() || !c.elem(i).rows()==1)
            {
                /* SG_ERROR("Expected String of type " error_string " as argument %d.\n", m_rhs_counter);*/
                SWIG_fail;
            }
            oct_type str=c.elem(i).oct_converter();

            INT len=str.cols();
            if (len>0) 
            { 
                strings[i].length=len; /* all must have same length in octave */
                strings[i].string=new sg_type[len+1]; /* not zero terminated in octave */
                /*ASSERT(strings[i].string);*/
                INT j; 
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
        INT len=data.rows();
        strings=new T_STRING<sg_type>[num_strings];
        ASSERT(strings);

        for (INT i=0; i<num_strings; i++)
        { 
            if (len>0) 
            { 
                strings[i].length=len; /* all must have same length in octave */
                strings[i].string=new sg_type[len+1]; /* not zero terminated in octave */
                /*ASSERT(strings[i].string);*/
                INT j;
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

GET_STRINGLIST(is_matrix_type() && arg.is_uint8_type, uint8NDArray, uint8_array_value, BYTE, BYTE, "Byte")
GET_STRINGLIST(is_char_matrix, charMatrix, char_matrix_value, char, char, "Char")
GET_STRINGLIST(is_matrix_type() && arg.is_int32_type, int32NDArray, int32_array_value, INT, INT, "Integer")
GET_STRINGLIST(is_matrix_type() && arg.is_int16_type, int16NDArray, int16_array_value, SHORT, SHORT, "Short")
GET_STRINGLIST(is_matrix_type() && arg.is_uint16_type, uint16NDArray, uint16_array_value, WORD, WORD, "Word")
#undef GET_STRINGLIST
