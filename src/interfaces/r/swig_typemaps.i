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

#include <shogun/lib/DataType.h>
#include <shogun/lib/memory.h>

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

/* One dimensional input arrays */
%define TYPEMAP_IN_SGVECTOR(r_type, r_cast, sg_type, error_string)
%typemap(typecheck, precedence=SWIG_TYPECHECK_POINTER)
    shogun::SGVector<sg_type>
{
    $1 = (TYPEOF($input) == r_type && Rf_ncols($input)==1 ) ? 1 : 0;
}

%typemap(in) shogun::SGVector<sg_type>
{
    SEXP rvec=$input;
    if (TYPEOF(rvec) != r_type || Rf_ncols(rvec)!=1)
    {
        /*SG_ERROR("Expected Double Vector as argument %d\n", m_rhs_counter);*/
        SWIG_fail;
    }

    $1 = shogun::SGVector<sg_type>((sg_type*) get_copy(r_cast(rvec), sizeof(sg_type)*LENGTH(rvec)), LENGTH(rvec));
}
%typemap(freearg) shogun::SGVector<sg_type>
{
}
%enddef

TYPEMAP_IN_SGVECTOR(INTSXP, INTEGER, int32_t, "Integer")
TYPEMAP_IN_SGVECTOR(REALSXP, REAL, float64_t, "Double Precision")
#undef TYPEMAP_IN_SGVECTOR

/* One dimensional output arrays */
%define TYPEMAP_OUT_SGVECTOR(r_type, r_cast, r_type_string, sg_type, if_type, error_string)
%typemap(out) shogun::SGVector<sg_type>
{
    sg_type* vec = $1.vector;
    auto len = $1.vlen;

    Rf_protect( $result = Rf_allocVector(r_type, len) );

    for (auto i=0; i<len; i++)
        r_cast($result)[i]=(if_type) vec[i];

    Rf_unprotect(1);
}

%typemap("rtype") shogun::SGVector<sg_type>   r_type_string

%typemap("scoerceout") shogun::SGVector<sg_type>
%{ %}

%enddef

TYPEMAP_OUT_SGVECTOR(INTSXP, INTEGER, "integer", uint8_t, int, "Byte")
TYPEMAP_OUT_SGVECTOR(INTSXP, INTEGER, "integer", int32_t, int, "Integer")
TYPEMAP_OUT_SGVECTOR(INTSXP, INTEGER, "integer", int16_t, int, "Short")
TYPEMAP_OUT_SGVECTOR(REALSXP, REAL, "numeric", float32_t, float, "Single Precision")
TYPEMAP_OUT_SGVECTOR(REALSXP, REAL, "numeric", float64_t, double, "Double Precision")
TYPEMAP_OUT_SGVECTOR(INTSXP, INTEGER, "integer", uint16_t, int, "Word")

#undef TYPEMAP_OUT_SGVECTOR

%define TYPEMAP_IN_SGMATRIX(r_type, r_cast, sg_type, error_string)
%typemap(typecheck, precedence=SWIG_TYPECHECK_POINTER)
        shogun::SGMatrix<sg_type>
{

    $1 = (TYPEOF($input) == r_type) ? 1 : 0;
}

%typemap(in) shogun::SGMatrix<sg_type>
{
    if( TYPEOF($input) != r_type)
    {
        /*SG_ERROR("Expected Double Matrix as argument %d\n", m_rhs_counter);*/
        SWIG_fail;
    }

    $1 = shogun::SGMatrix<sg_type>((sg_type*) get_copy(r_cast($input), ((size_t) Rf_nrows($input))*Rf_ncols($input)*sizeof(sg_type)), Rf_nrows($input), Rf_ncols($input));
}
%typemap(freearg) shogun::SGMatrix<sg_type>
{
}
%enddef

TYPEMAP_IN_SGMATRIX(INTSXP, INTEGER, int32_t, "Integer")
TYPEMAP_IN_SGMATRIX(REALSXP, REAL, float64_t, "Double Precision")
#undef TYPEMAP_IN_SGMATRIX

%define TYPEMAP_OUT_SGMATRIX(r_type, r_cast, sg_type, if_type, error_string)
%typemap(out) shogun::SGMatrix<sg_type>
{
    sg_type* matrix = $1.matrix;
    auto num_feat = $1.num_rows;
    auto num_vec = $1.num_cols;

    Rf_protect( $result = Rf_allocMatrix(r_type, num_feat, num_vec) );

    for (auto i=0; i<num_vec; i++)
    {
        for (auto j=0; j<num_feat; j++)
            r_cast($result)[i*num_feat+j]=(if_type) matrix[i*num_feat+j];
    }

    Rf_unprotect(1);
}

%typemap("rtype") shogun::SGMatrix<sg_type>   "matrix"

%typemap("scoerceout") shogun::SGMatrix<sg_type>
%{ %}

%enddef

TYPEMAP_OUT_SGMATRIX(INTSXP, INTEGER, uint8_t, int, "Byte")
TYPEMAP_OUT_SGMATRIX(INTSXP, INTEGER, int32_t, int, "Integer")
TYPEMAP_OUT_SGMATRIX(INTSXP, INTEGER, int16_t, int, "Short")
TYPEMAP_OUT_SGMATRIX(REALSXP, REAL, float32_t, float, "Single Precision")
TYPEMAP_OUT_SGMATRIX(REALSXP, REAL, float64_t, double, "Double Precision")
TYPEMAP_OUT_SGMATRIX(INTSXP, INTEGER, uint16_t, int, "Word")
#undef TYPEMAP_OUT_SGMATRIX

/* TODO INND ARRAYS */

/* input typemap for CStringFeatures<char> etc */
%define TYPEMAP_STRINGFEATURES_IN(r_type, sg_type, if_type, error_string)
%typemap(in) shogun::SGStringList<sg_type>
{
    auto max_len=0;
    auto num_strings=0;
    shogun::SGString<sg_type>* strs=NULL;

    if ($input == R_NilValue || TYPEOF($input) != STRSXP)
    {
        /* SG_ERROR("Expected String List as argument %d\n", m_rhs_counter);*/
        SWIG_fail;
    }

    num_strings=Rf_length($input);
    ASSERT(num_strings>=1);
    strs=SG_MALLOC(shogun::SGString<sg_type>, num_strings);

    for (auto i=0; i<num_strings; i++)
    {
        SEXPREC* s= STRING_ELT($input,i);
        sg_type* c= (sg_type*) if_type(s);
        auto len=LENGTH(s);

        if (len>0)
        {
			sg_type* dst=SG_MALLOC(sg_type, len+1);
            /*ASSERT(strs[i].string);*/
			strs[i].string=(sg_type*) sg_memcpy(dst, c, len*sizeof(sg_type));
            strs[i].string[len]='\0'; /* zero terminate */
            strs[i].slen=len;
            max_len=CMath::max(max_len, len);
        }
        else
        {
            /*SG_WARNING( "string with index %d has zero length.\n", i+1);*/
            strs[i].slen=0;
            strs[i].string=NULL;
        }
    }

    SGStringList<sg_type> sl;
    sl.strings=strs;
    sl.num_strings=num_strings;
    sl.max_string_length=max_len;
    $1 = sl;
}
%enddef

TYPEMAP_STRINGFEATURES_IN(STRSXP, char, CHAR, "Char")
#undef TYPEMAP_STRINGFEATURES_IN

/* TODO STRING OUT TYPEMAPS */
