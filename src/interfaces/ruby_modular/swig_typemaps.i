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
	$1 = (
					($input && TYPE($input) == T_ARRAY && RARRAY_LEN($input) > 0) ||
					($input && NA_IsNArray($input) && NA_SHAPE0($input) > 0)
			)? 1 : 0;
}

%typemap(in) shogun::SGVector<SGTYPE> {
	int32_t i, len;
	SGTYPE *array;
	VALUE *ptr;

	if (rb_obj_is_kind_of($input,rb_cArray)) {
		len = RARRAY_LEN($input);
		array = SG_MALLOC(SGTYPE, len);

		ptr = RARRAY_PTR($input);
		for (i = 0; i < len; i++, ptr++) {
			array[i] = R2SG(*ptr);
		}
	}
	else {
		if (NA_IsNArray($input)) {

			VALUE v = (*na_to_array_dl)($input);
			len = RARRAY_LEN(v);
			array = SG_MALLOC(SGTYPE, len);

			ptr = RARRAY_PTR(v);
			for (i = 0; i < len; i++, ptr++) {
				array[i] = R2SG(*ptr);
			}
		}
		else {
			rb_raise(rb_eArgError, "Expected Array");
		}
	}

	$1 = shogun::SGVector<SGTYPE>((SGTYPE *)array, len);
}

%typemap(out) shogun::SGVector<SGTYPE> {
	int32_t i;
	VALUE arr = rb_ary_new2($1.vlen);

	for (i = 0; i < $1.vlen; i++)
		rb_ary_push(arr, SG2R($1.vector[i]));

	$result = (*na_to_narray_dl)(arr);
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

/* Two dimensional input/output arrays */
%define TYPEMAP_SGMATRIX(SGTYPE, R2SG, SG2R)

%typemap(typecheck, precedence=SWIG_TYPECHECK_POINTER) shogun::SGMatrix<SGTYPE>
{
	$1 = (
					($input && TYPE($input) == T_ARRAY && RARRAY_LEN($input) > 0 && TYPE(rb_ary_entry($input, 0)) == T_ARRAY) ||
					($input &&  NA_IsNArray($input) && NA_SHAPE1($input) > 0 && NA_SHAPE0($input) > 0)		
				) ? 1 : 0;
}

%typemap(in) shogun::SGMatrix<SGTYPE> {
	int32_t i, j, rows, cols;
	SGTYPE *array;
	VALUE vec;
	VALUE v;

	if (rb_obj_is_kind_of($input,rb_cArray) || NA_IsNArray($input)) {
		if (NA_IsNArray($input))	{
			v = (*na_to_array_dl)($input);	
		}
		else {
			v = $input;		
		}
		rows = RARRAY_LEN(v);
		cols = 0;

		for (i = 0; i < rows; i++) {
			vec = rb_ary_entry(v, i);
			if (!rb_obj_is_kind_of(vec,rb_cArray)) {
				rb_raise(rb_eArgError, "Expected Arrays");
			}
			if (cols == 0) {
				cols = RARRAY_LEN(vec);
				array = SG_MALLOC(SGTYPE, rows * cols);
			}
			for (j = 0; j < cols; j++) {
				array[i * cols + j] = R2SG(rb_ary_entry(vec, j));
			}
		}
	}
	else {
		rb_raise(rb_eArgError, "Expected Arrays");
	}
	$1 = shogun::SGMatrix<SGTYPE>((SGTYPE*)array, rows, cols, true);
}

%typemap(out) shogun::SGMatrix<SGTYPE> {
	int32_t rows = $1.num_rows;
	int32_t cols = $1.num_cols;
	int32_t len = rows * cols;
	VALUE arr;
	int32_t i, j;
	
	arr = rb_ary_new2(rows);
		
	for (i = 0; i < rows; i++) {
		VALUE vec = rb_ary_new2(cols);
		for (j = 0; j < cols; j++) {
			rb_ary_push(vec, SG2R($1.matrix[i * cols + j]));
		}
		rb_ary_push(arr, vec);
	}

	$result = (*na_to_narray_dl)(arr);
}

%enddef

/* Define concrete examples of the TYPEMAP_SGMATRIX macros */
TYPEMAP_SGMATRIX(char, NUM2CHR, CHR2FIX)
TYPEMAP_SGMATRIX(uint16_t, NUM2INT, INT2NUM)
TYPEMAP_SGMATRIX(int32_t, NUM2INT, INT2NUM)
TYPEMAP_SGMATRIX(uint32_t, NUM2UINT, UINT2NUM)
TYPEMAP_SGMATRIX(int64_t, NUM2LONG,  LONG2NUM)
TYPEMAP_SGMATRIX(uint64_t, NUM2ULONG, ULONG2NUM)
TYPEMAP_SGMATRIX(long long, NUM2LL, LL2NUM)
TYPEMAP_SGMATRIX(float32_t, NUM2DBL, rb_float_new)
TYPEMAP_SGMATRIX(float64_t, NUM2DBL, rb_float_new)

#undef TYPEMAP_SGMATRIX

/* input/output typemap for CStringFeatures */
%define TYPEMAP_STRINGFEATURES(SGTYPE, R2SG, SG2R, TYPECODE)

%typemap(typecheck, precedence=SWIG_TYPECHECK_POINTER) shogun::SGStringList<SGTYPE> {
	$1 = 0;
	if (TYPE($input) == T_ARRAY && RARRAY_LEN($input) > 0) {
		$1 = 1;
	}
}
%typemap(in) shogun::SGStringList<SGTYPE> {
	int32_t size = 0;
	int32_t i, j;
	int32_t len, max_len = 0;

	if (TYPE($input) != T_ARRAY) {
		rb_raise(rb_eArgError, "Expected Arrays");
	}
	
	size = RARRAY_LEN($input);
	shogun::SGString<SGTYPE>* strings=SG_MALLOC(shogun::SGString<SGTYPE>, size);

	for (i = 0; i < size; i++) {
		VALUE arr = rb_ary_entry($input, i);
		new (&strings[i]) SGString<SGTYPE>();
		if (TYPE(arr) == T_STRING) {
			len = RSTRING_LEN(arr);
			const char *str = StringValuePtr(arr);
			max_len = shogun::CMath::max(len, max_len);

			strings[i].slen = len;
			strings[i].string = NULL;
			
			if (len > 0) {
				strings[i].string = SG_MALLOC(SGTYPE, len + 1);
				memcpy(strings[i].string, str, len + 1);
			}
		}
		else {
			if (TYPE(arr) == T_ARRAY) {
				len = RARRAY_LEN(arr);
				max_len = shogun::CMath::max(len, max_len);

				strings[i].slen=len;
				strings[i].string=NULL;
				if (len > 0) {
					strings[i].string = SG_MALLOC(SGTYPE, len);
					for (j = 0; j < len; j++) {
						strings[i].string[j] = R2SG(RARRAY_PTR(arr)[j]);
					}
				}
			}
			else {
				rb_raise(rb_eArgError, "Expected Arrays");
			}
		}
	}
	
	SGStringList<SGTYPE> sl;
	sl.strings = strings;
	sl.num_strings = size;
	sl.max_string_length = max_len;
	$1 = sl;
}

%typemap(out) shogun::SGStringList<SGTYPE> {
	shogun::SGString<SGTYPE>* str = $1.strings;
	int32_t i, j, num = $1.num_strings;
	VALUE arr;

	arr = rb_ary_new2(num);

	for (i = 0; i < num; i++) {
		if (strcmp(TYPECODE, "String[]")==0) {
			VALUE vec = rb_str_new2((char *)str[i].string);
			rb_ary_push(arr, vec);
		}
		else {
			SGTYPE* data = SG_MALLOC(SGTYPE, str[i].slen);
			memcpy(data, str[i].string, str[i].slen * sizeof(SGTYPE));
			
			VALUE vec = rb_ary_new2(str[i].slen);
			for (j = 0; j < str[i].slen; j++) {
				rb_ary_push(vec, SG2R(data[j]));
			}
			rb_ary_push(arr, vec);
		}
	}
	$result = arr;
}

%enddef

TYPEMAP_STRINGFEATURES(char, NUM2CHR, CHR2FIX, "String[]")
TYPEMAP_STRINGFEATURES(uint16_t, NUM2INT, INT2NUM, "uint16_t[][]")
TYPEMAP_STRINGFEATURES(int32_t, NUM2INT, INT2NUM, "int32_t[][]")
TYPEMAP_STRINGFEATURES(uint32_t, NUM2UINT, UINT2NUM, "uint32_t[][]")
TYPEMAP_STRINGFEATURES(int64_t, NUM2LONG,  LONG2NUM, "int64_t[][]")
TYPEMAP_STRINGFEATURES(uint64_t, NUM2ULONG, ULONG2NUM, "uint64_t[][]")
TYPEMAP_STRINGFEATURES(long long, NUM2LL, LL2NUM, "long long[][]")
TYPEMAP_STRINGFEATURES(float32_t, NUM2DBL, rb_float_new, "float32_t[][]")
TYPEMAP_STRINGFEATURES(float64_t, NUM2DBL, rb_float_new, "float64_t[][]")

#undef TYPEMAP_STRINGFEATURES
