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
extern "C" {
#include <R.h>
#include <Rinternals.h>
#include <Rdefines.h>
#include <R_ext/Rdynload.h>
#include <Rembedded.h>
#include <Rinterface.h>
#include <R_ext/RS.h>
#include <R_ext/Error.h>
}

/* workaround compile bug in R-modular interface */
#ifndef ScalarReal
#define ScalarReal      Rf_ScalarReal
#endif
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
%define TYPEMAP_IN1(r_type, r_cast, sg_type, error_string)
%typemap(typecheck, precedence=SWIG_TYPECHECK_POINTER)
    (sg_type* IN_ARRAY1, int32_t DIM1)
{
    $1 = (TYPEOF($input) == r_type && Rf_ncols($input)==1 ) ? 1 : 0;
}

%typemap(in) (sg_type* IN_ARRAY1, int32_t DIM1) (SEXP rvec)
{
    rvec=$input;
    if (TYPEOF(rvec) != r_type || Rf_ncols(rvec)!=1)
    {
        /*SG_ERROR("Expected Double Vector as argument %d\n", m_rhs_counter);*/
        SWIG_fail;
    }

    $1 = (sg_type*) r_cast(rvec);
    $2 = LENGTH(rvec);
}
%typemap(freearg) (type* IN_ARRAY1, int32_t DIM1) {
}
%enddef

TYPEMAP_IN1(INTSXP, INTEGER, int32_t, "Integer")
TYPEMAP_IN1(REALSXP, REAL, float64_t, "Double Precision")
#undef TYPEMAP_IN1

%define TYPEMAP_IN2(r_type, r_cast, sg_type, error_string)
%typemap(typecheck, precedence=SWIG_TYPECHECK_POINTER)
        (sg_type* IN_ARRAY2, int32_t DIM1, int32_t DIM2)
{

    $1 = (TYPEOF($input) == r_type) ? 1 : 0;
}

%typemap(in) (sg_type* IN_ARRAY2, int32_t DIM1, int32_t DIM2)
{
    if( TYPEOF($input) != r_type)
    {
        /*SG_ERROR("Expected Double Matrix as argument %d\n", m_rhs_counter);*/
        SWIG_fail;
    }

    $1 = (sg_type*) r_cast($input);
    $2 = Rf_nrows($input);
    $3 = Rf_ncols($input);
}
%typemap(freearg) (type* IN_ARRAY2, int32_t DIM1, int32_t DIM2) {
}
%enddef

TYPEMAP_IN2(INTSXP, INTEGER, int32_t, "Integer")
TYPEMAP_IN2(REALSXP, REAL, float64_t, "Double Precision")
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
%define TYPEMAP_ARGOUT1(r_type, r_cast, sg_type, if_type, error_string)
%typemap(in, numinputs=0) (sg_type** ARGOUT1, int32_t* DIM1) {
    $1 = (sg_type**) malloc(sizeof(sg_type*));
    $2 = (int32_t*) malloc(sizeof(int32_t));
}

%typemap(argout) (sg_type** ARGOUT1, int32_t* DIM1) {
    sg_type* vec = *$1;
    int32_t len = *$2;

    Rf_protect( $result = Rf_allocVector(r_type, len) );

    for (int32_t i=0; i<len; i++)
        r_cast($result)[i]=(if_type) vec[i];

    Rf_unprotect(1);
    free(*$1); free($1); free($2);
}
%enddef

TYPEMAP_ARGOUT1(INTSXP, INTEGER, uint8_t, int, "Byte")
TYPEMAP_ARGOUT1(INTSXP, INTEGER, int32_t, int, "Integer")
TYPEMAP_ARGOUT1(INTSXP, INTEGER, int16_t, int, "Short")
TYPEMAP_ARGOUT1(REALSXP, REAL, float32_t, float, "Single Precision")
TYPEMAP_ARGOUT1(REALSXP, REAL, float64_t, double, "Double Precision")
TYPEMAP_ARGOUT1(INTSXP, INTEGER, uint16_t, int, "Word")
#undef TYPEMAP_ARGOUT1

%define TYPEMAP_ARGOUT2(r_type, r_cast, sg_type, if_type, error_string)
%typemap(in, numinputs=0) (sg_type** ARGOUT2, int32_t* DIM1, int32_t* DIM2) {
    $1 = (sg_type**) malloc(sizeof(sg_type*));
    $2 = (int32_t*) malloc(sizeof(int32_t));
    $3 = (int32_t*) malloc(sizeof(int32_t));
}

%typemap(argout) (sg_type** ARGOUT2, int32_t* DIM1, int32_t* DIM2) {
    sg_type* matrix = *$1;
    int32_t num_feat = *$2;
    int32_t num_vec = *$3;

    Rf_protect( $result = Rf_allocMatrix(r_type, num_feat, num_vec) );

    for (int32_t i=0; i<num_vec; i++)
    {
        for (int32_t j=0; j<num_feat; j++)
            r_cast($result)[i*num_feat+j]=(if_type) matrix[i*num_feat+j];
    }
    Rf_unprotect(1);
    free(*$1); free($1); free($2); free($3);
}
%enddef

TYPEMAP_ARGOUT2(INTSXP, INTEGER, uint8_t, int, "Byte")
TYPEMAP_ARGOUT2(INTSXP, INTEGER, int32_t, int, "Integer")
TYPEMAP_ARGOUT2(INTSXP, INTEGER, int16_t, int, "Short")
TYPEMAP_ARGOUT2(REALSXP, REAL, float32_t, float, "Single Precision")
TYPEMAP_ARGOUT2(REALSXP, REAL, float64_t, double, "Double Precision")
TYPEMAP_ARGOUT2(INTSXP, INTEGER, uint16_t, int, "Word")
#undef TYPEMAP_ARGOUT2

/* input typemap for CStringFeatures<char> etc */
%define TYPEMAP_STRINGFEATURES_IN(r_type, sg_type, if_type, error_string)
%typemap(in) (shogun::SGString<sg_type>* IN_STRINGS, int32_t NUM, int32_t MAXLEN)
{
    int32_t max_len=0;
    int32_t num_strings=0;
    shogun::SGString<sg_type>* strs=NULL;

    if ($input == R_NilValue || TYPEOF($input) != STRSXP)
    {
        /* SG_ERROR("Expected String List as argument %d\n", m_rhs_counter);*/
        SWIG_fail;
    }

    num_strings=Rf_length($input);
    ASSERT(num_strings>=1);
    strs=new shogun::SGString<sg_type>[num_strings];

    for (int32_t i=0; i<num_strings; i++)
    {
        SEXPREC* s= STRING_ELT($input,i);
        sg_type* c= (sg_type*) if_type(s);
        int32_t len=LENGTH(s);

        if (len>0) 
        { 
			sg_type* dst=new sg_type[len+1];
            /*ASSERT(strs[i].string);*/
			strs[i].string=(sg_type*) memcpy(dst, c, len*sizeof(sg_type));
            strs[i].string[len]='\0'; /* zero terminate */
            strs[i].length=len;
            max_len=CMath::max(max_len, len);
        }
        else
        {
            /*SG_WARNING( "string with index %d has zero length.\n", i+1);*/
            strs[i].length=0;
            strs[i].string=NULL;
        }
    }
    $1 = strs;
    $2 = num_strings;
    $3 = max_len;
}
%enddef

TYPEMAP_STRINGFEATURES_IN(STRSXP, char, CHAR, "Char")
#undef TYPEMAP_STRINGFEATURES_IN
