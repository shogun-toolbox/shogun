/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Viktor Gal, Heiko Strathmann, Sergey Lisitsyn, Elias Saalmann
 */

/* One dimensional input/output arrays */
%define TYPEMAP_SGVECTOR(SGTYPE, R2SG, SG2R)

%typemap(typecheck, precedence=SWIG_TYPECHECK_POINTER) shogun::SGVector<SGTYPE> {
	$1 = (
					($input && TYPE($input) == T_ARRAY && RARRAY_LEN($input) > 0) ||
					($input && NM_IsNMatrix($input) && NM_DIM($input) == 1 && NM_SHAPE0($input) > 0)
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
                if (NM_IsNMatrix($input) && NM_DIM($input) == 1) {
                        len = NM_SHAPE0($input);
                        array = SG_MALLOC(SGTYPE, len);

                        // FIXME: there will be dragons
                        // NM_SIZEOF_DTYPE has no ref... hence the hack
                        void* p = NM_DENSE_ELEMENTS($input);
                        memcpy(array, p, len*sizeof(SGTYPE));
                }
                else {
                        rb_raise(rb_eArgError, "Expected Array");
                }
        }

        $1 = shogun::SGVector<SGTYPE>((SGTYPE *)array, len);
}

%typemap(out) shogun::SGVector<SGTYPE> {
	int32_t i;
        $result = rb_nvector_dense_create(SG2R, static_cast<void*>($1.vector), $1.vlen);
}

%enddef

/* Define concrete examples of the TYPEMAP_SGVECTOR macros */
/* FIXME: the dtype_t are not really correct in unsigned cases  */
TYPEMAP_SGVECTOR(char, NUM2CHR, nm::INT8)
TYPEMAP_SGVECTOR(uint16_t, NUM2INT, nm::INT16)
TYPEMAP_SGVECTOR(int32_t, NUM2INT, nm::INT32)
TYPEMAP_SGVECTOR(uint32_t, NUM2UINT, nm::INT32)
TYPEMAP_SGVECTOR(int64_t, NUM2LONG,  nm::INT64)
TYPEMAP_SGVECTOR(uint64_t, NUM2ULONG, nm::INT64)
TYPEMAP_SGVECTOR(long long, NUM2LL, nm::INT64)
TYPEMAP_SGVECTOR(float32_t, NUM2DBL, nm::FLOAT32)
TYPEMAP_SGVECTOR(float64_t, NUM2DBL, nm::FLOAT64)

#undef TYPEMAP_SGVECTOR

/* Two dimensional input/output arrays */
%define TYPEMAP_SGMATRIX(SGTYPE, R2SG, SG2R)

%typemap(typecheck, precedence=SWIG_TYPECHECK_POINTER) shogun::SGMatrix<SGTYPE>
{
	$1 = (
					($input && TYPE($input) == T_ARRAY && RARRAY_LEN($input) > 0 && TYPE(rb_ary_entry($input, 0)) == T_ARRAY) ||
					($input &&  NM_IsNMatrix($input) && NM_DIM($input) == 2 && NM_SHAPE1($input) > 0 && NM_SHAPE0($input) > 0)
				) ? 1 : 0;
}

%typemap(in) shogun::SGMatrix<SGTYPE> {
        int32_t i, j, rows, cols;
        SGTYPE *array;
        VALUE vec;
        VALUE v;

        if (rb_obj_is_kind_of($input,rb_cArray) || (NM_IsNMatrix($input) && NM_DIM($input) == 2)) {
                if (NM_IsNMatrix($input)) {
                        if (NM_STYPE($input) == nm::DENSE_STORE) {
                                rows = NM_SHAPE0($input);
                                cols = NM_SHAPE1($input);
                                array = SG_MALLOC(SGTYPE, rows * cols);

                                // FIXME: this cast is not correct in case of unsigned types...
                                auto *ptr = static_cast<SGTYPE*>(NM_DENSE_ELEMENTS($input));
                                for (i = 0; i < rows; ++i) {
                                        for (j = 0; j < cols; ++j) {
                                                array[j * rows + i] = ptr[i * cols + j];
                                        }
                                }
                        } else {
                                //TODO: handle other storage types
                                rb_raise(rb_eArgError, "Expected nm::DENSE NMatrix");
                        }
                } else {
                        v = $input;
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
                                        array[j * rows + i] = R2SG(rb_ary_entry(vec, j));
                                }
                        }
                }
        }
        else {
                rb_raise(rb_eArgError, "Expected Arrays");
        }
        $1 = shogun::SGMatrix<SGTYPE>((SGTYPE*)array, rows, cols, true);
}

%typemap(out) shogun::SGMatrix<SGTYPE> {
        size_t shape[2];
        shape[0] = $1.num_rows;
        shape[1] = $1.num_cols;
        size_t len = shape[0] * shape[1];

        // FIMXE: this needs to be transposed!!
        $result = rb_nmatrix_dense_create(SG2R, shape, 2, static_cast<void*>($1.matrix), len);
}

%enddef

/* Define concrete examples of the TYPEMAP_SGMATRIX macros */
TYPEMAP_SGMATRIX(char, NUM2CHR, nm::INT8)
TYPEMAP_SGMATRIX(uint16_t, NUM2INT, nm::INT16)
TYPEMAP_SGMATRIX(int32_t, NUM2INT, nm::INT32)
TYPEMAP_SGMATRIX(uint32_t, NUM2UINT, nm::INT32)
TYPEMAP_SGMATRIX(int64_t, NUM2LONG,  nm::INT64)
TYPEMAP_SGMATRIX(uint64_t, NUM2ULONG, nm::INT64)
TYPEMAP_SGMATRIX(long long, NUM2LL, nm::INT64)
TYPEMAP_SGMATRIX(float32_t, NUM2DBL, nm::FLOAT32)
TYPEMAP_SGMATRIX(float64_t, NUM2DBL, nm::FLOAT64)

#undef TYPEMAP_SGMATRIX

/* input/output typemap for CStringFeatures */
%define TYPEMAP_STRINGFEATURES(SGTYPE, R2SG, SG2R, TYPECODE)

%typemap(typecheck, precedence=SWIG_TYPECHECK_POINTER) std::vector<shogun::SGVector<SGTYPE>> {
        $1 = 0;
        if (TYPE($input) == T_ARRAY && RARRAY_LEN($input) > 0) {
                $1 = 1;
        }
}
%typemap(in) std::vector<shogun::SGVector<SGTYPE>> {
	std::vector<shogun::SGVector<SGTYPE>>& strings = $1;

	int32_t size = 0;
	int32_t i, j;
	int32_t len, max_len = 0;

	if (TYPE($input) != T_ARRAY) {
		rb_raise(rb_eArgError, "Expected Arrays");
	}

	size = RARRAY_LEN($input);
	strings.reserve(size);

	for (i = 0; i < size; i++) {
		VALUE arr = rb_ary_entry($input, i);
		if (TYPE(arr) == T_STRING) {
			len = RSTRING_LEN(arr);
			strings.emplace_back(len+1);
			strings.back().vlen = len;

			const char *str = StringValuePtr(arr);
			max_len = shogun::Math::max(len, max_len);

			sg_memcpy(strings.back().vector, str, len + 1);
		}
		else {
			if (TYPE(arr) == T_ARRAY) {
				len = RARRAY_LEN(arr);
				max_len = shogun::Math::max(len, max_len);
				strings.emplace_back(len);

				for (j = 0; j < len; j++) {
					strings.back().vector[j] = R2SG(RARRAY_PTR(arr)[j]);
				}
			}
			else {
				rb_raise(rb_eArgError, "Expected Arrays");
			}
		}
	}
}

%typemap(out) std::vector<shogun::SGVector<SGTYPE>> {
	std::vector<shogun::SGVector<SGTYPE>>& str = $1;
	int32_t i, j, num = str.size();
	VALUE arr;

	arr = rb_ary_new2(num);

	for (i = 0; i < num; i++) {
		if (strcmp(TYPECODE, "String[]")==0) {
			VALUE vec = rb_str_new2((char *)str[i].vector);
			rb_ary_push(arr, vec);
		}
		else {
			SGTYPE* data = SG_MALLOC(SGTYPE, str[i].vlen);
			sg_memcpy(data, str[i].vector, str[i].vlen * sizeof(SGTYPE));

			VALUE vec = rb_ary_new2(str[i].vlen);
			for (j = 0; j < str[i].vlen; j++) {
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
