/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
  *
 * Written (W) 2011 Baozeng Ding
  *  
 */

/* One dimensional input/output arrays */
%define TYPEMAP_SGVECTOR(SGTYPE, R2SG, SG2R)

%typemap(typecheck, precedence=SWIG_TYPECHECK_POINTER) shogun::SGVector<SGTYPE> {
	$1 = ($input && TYPE($input) == T_ARRAY && RARRAY($input)->len > 0) ? 1 : 0;
	
}

%typemap(in) shogun::SGVector<SGTYPE> {
	int32_t i, len;
	SGTYPE *array;
	VALUE *ptr;
	
	if (!rb_obj_is_kind_of($input,rb_cArray))
		rb_raise(rb_eArgError, "Expected Array");

	len = RARRAY($input)->len;
	array = new SGTYPE[len];
	
	ptr = RARRAY($input)->ptr;
	for (i = 0; i < len; i++, ptr++) {
		array[i] = R2SG(*ptr);
	}
	
	$1 = shogun::SGVector<SGTYPE>((SGTYPE *)array, len);
}

%typemap(out) shogun::SGVector<SGTYPE> {
	int32_t i;	
	VALUE arr = rb_ary_new2($1.length);
		
	for (i = 0; i < $1.length; i++)
		rb_ary_push(arr, SG2R($1.vector[i]));

	$result = arr;
}

%enddef

/* Define concrete examples of the TYPEMAP_SGVECTOR macros */
TYPEMAP_SGVECTOR(char, NUM2CHR, CHR2FIX)
TYPEMAP_SGVECTOR(uint16_t, NUM2INT, INT2NUM)
TYPEMAP_SGVECTOR(int32_t, NUM2INT, INT2NUM)
TYPEMAP_SGVECTOR(uint32_t, NUM2UINT, UINT2NUM)
TYPEMAP_SGVECTOR(int64_t, NUM2LONG,  LONG2NUM)
TYPEMAP_SGVECTOR(uint64_t, NUM2ULONG, ULONG2NUM)
TYPEMAP_SGVECTOR(long long, NUM2LL, LL2NUM)
TYPEMAP_SGVECTOR(float32_t, NUM2DBL, rb_float_new)
TYPEMAP_SGVECTOR(float64_t, NUM2DBL, rb_float_new)

#undef TYPEMAP_SGVECTOR
