/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Copyright (C) 2012 Evgeniy Andreev (gsomix)
 */

/* helper's stuff */

%define BUFFER_VECTOR_INFO(type_name)
%header
%{
  //PTZ121004 really shall be linked to pdl type indeed.
typedef
struct buffer_vector_ ## type_name ## _info
{
	SGVector< type_name > buf; 
	STRLEN* shape;
	STRLEN* strides;
	void* internal;
} buffer_vector_ ## type_name ## _info;

%}
%enddef // BUFFER_VECTOR_INFO

%define BUFFER_MATRIX_INFO(type_name)
%header
%{

struct buffer_matrix_ ## type_name ## _info
{
	SGMatrix< type_name >  buf; 
	STRLEN* shape;
	STRLEN* strides;
	void* internal;
};

%}
%enddef // BUFFER_MATRIX_INFO

BUFFER_VECTOR_INFO(bool)
BUFFER_VECTOR_INFO(char)
BUFFER_VECTOR_INFO(uint8_t)
BUFFER_VECTOR_INFO(uint16_t)
BUFFER_VECTOR_INFO(int16_t)
BUFFER_VECTOR_INFO(int32_t)
BUFFER_VECTOR_INFO(uint32_t)
BUFFER_VECTOR_INFO(int64_t)
BUFFER_VECTOR_INFO(uint64_t)
BUFFER_VECTOR_INFO(float32_t)
BUFFER_VECTOR_INFO(float64_t)

BUFFER_MATRIX_INFO(bool)
BUFFER_MATRIX_INFO(char)
BUFFER_MATRIX_INFO(uint8_t)
BUFFER_MATRIX_INFO(uint16_t)
BUFFER_MATRIX_INFO(int16_t)
BUFFER_MATRIX_INFO(int32_t)
BUFFER_MATRIX_INFO(uint32_t)
BUFFER_MATRIX_INFO(int64_t)
BUFFER_MATRIX_INFO(uint64_t)
BUFFER_MATRIX_INFO(float32_t)
BUFFER_MATRIX_INFO(float64_t)


%wrapper
%{
#include <cstring>

  void get_slice_in_bounds(STRLEN* ilow, STRLEN* ihigh, STRLEN max_idx)
  {
    if (*ilow<0)
      {
	*ilow=0;
      }
    else if (*ilow>max_idx)
      {
	*ilow = max_idx;
      }
    if (*ihigh<*ilow)
      {
	*ihigh=*ilow;
      }
    else if (*ihigh>max_idx)
      {
	*ihigh=max_idx;
      }
  }

  STRLEN get_idx_in_bounds(STRLEN idx, STRLEN max_idx)
  {
    if (idx >= max_idx || idx < (-max_idx))
      {
	pdl_warn("index out of bounds"); //SWIG_error, pdl_warn ::warn
	return -1;
      }
    else if (idx < 0)
      return idx + max_idx;
    return idx;
  }


  SWIGRUNTIME
  int parse_tuple_item(SWIG_MAYBE_PERL_OBJECT SV* item
		       , SSize_t length, SSize_t* ilow, SSize_t* ihigh,
		       SSize_t* step, SSize_t* slicelength)
  {
    //swig_cast_info *tc;
    //void *voidptr = (void *)0;
    SV *tsv = (SV*) SvRV(item);
    IV tmp = 0;

    /* If magical, apply more magic */
    if (SvGMAGICAL(item))
      mg_get(item);
    /* Check to see if this is an object */
    if (sv_isobject(item)) {
      if ((SvTYPE(tsv) == SVt_PVHV)) {
	MAGIC *mg;
	if (SvMAGICAL(tsv)) {
	  mg = mg_find(tsv,'P');
	  if (mg) {
	    item = mg->mg_obj;
	    if (sv_isobject(item)) {
	      tsv = (SV*)SvRV(item);
	      tmp = SvIV(tsv);
	    }
	  }
	} else {
	  return 0;//SWIG_ERROR;
	}
      } else {
	tmp = SvIV(tsv);
      }
    } else if (! SvOK(item)) {
      return 0;
    } else if (SvTYPE(item) == SVt_RV) {  /* Check for NULL pointer */
      if (!SvROK(item)) {
      /* In Perl 5.12 and later, SVt_RV == SVt_IV, so sv could be a valid integer value.  */
	if (SvIOK(item)) {
	  //return SWIG_ERROR;
	  //tmp = SvIV(item);
	  tmp = SvIV(tsv);
	} else {
	  /* NULL pointer (reference to undef). */
	  //*(ptr) = (void *) 0;
	  return 0;//SWIG_OK;
	}
      } else {
	return 0;//SWIG_ERROR;
      }
    } else {                            /* Don't know what it is */
      return 0;//SWIG_ERROR;
  }
#if PTZ120926_not_ready_yet_use_ranges
    if (
	PySlice_Check(item)
	)
      {
	PySlice_GetIndicesEx((PySliceObject*) item, length, ilow, ihigh, step, slicelength);
	get_slice_in_bounds(ilow, ihigh, length);
	
	return 2;
      }
    else
#endif
      // || PyArray_IsScalar(item, Integer)
      // || (PyIndex_Check(item) && !PySequence_Check(item))
      //STMT_START {
      //SV* const xsub_tmp_sv = item;
      //SvGETMAGIC(xsub_tmp_sv);
      //} STMT_END;
      //      if (
      //	  SvTYPE(SvRV(item)) == SVt_PVIV
      //	  || SvTYPE(SvRV(item)) == SVt_PVAV
      //	  )

	{
	  //PyArray_PyIntAsIntp
	  //PTZ120926 is this for getting the "current" key index of the AV item?
	  SSize_t idx = INT2PTR(SSize_t, tmp);
	  idx = get_idx_in_bounds(idx, length);
	
	  *ilow = idx;
	  *ihigh = idx + 1;
	
	  return 1;
	}
    return 0;
  }
  //PyMethodDef*
  //PyCFunction SV-RV-PVCV CODE
  //use SWIG!!!
  //  void set_method(PyMethodDef* methods, const char* name, PyCFunction new_method)
  //like... SWIG_TypeClientData(SWIGTYPE_p_shogun__CAlphabet, (void*) "modshogun::Alphabet");
  //    SWIG_TypeClientData(SWIGTYPE_p_shogun__CStochasticProximityEmbedding, (void*) "modshogun::StochasticProximityEmbedding");
/*@SWIG:/usr/share/swig2.0/perl5/perltypemaps.swg,65,%set_constant@*/
/*  do { */
/*     SV *sv = get_sv((char*) SWIG_prefix "P_UNKNOWN", TRUE | 0x2 | GV_ADDMULTI); */
/*     sv_setsv(sv, SWIG_From_int  SWIG_PERL_CALL_ARGS_1(static_cast< int >(shogun::P_UNKNOWN))); */
/*     SvREADONLY_on(sv); */
/*   } while(0) /\*@SWIG@*\/; */
//static swig_command_info swig_commands[] = {
/* SWIGRUNTIME void */
/* SWIG_InitializeModule(void *clientdata) { */
/*   size_t i; */
/*   swig_module_info *module_head, *iter; */
/*   int found, init; */
#ifdef PTZ120926_is_not_ready_for_overloading_perl_c_yeak
  void set_method(SV* methods, const char* name, SV* new_method)
  {
    //PyMethodDef method_temp;
    SV* method_temp;
    int method_idx = 0;
	
    do
      {
	method_temp = methods[method_idx];
	method_idx++;
      }
    while (strcmp(method_temp.ml_name, name) != 0 && method_temp.ml_name != NULL);
    
    methods[method_idx-1].ml_meth = new_method;
  }

#endif


%}
