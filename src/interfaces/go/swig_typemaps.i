/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Viktor Gal
 */

%define TYPEMAP_SGVECTOR(go_type, sg_type)

%typemap(gotype) shogun::SGVector<sg_type> %{[]go_type%}

%typemap(in) shogun::SGVector<sg_type>
{
	sg_type* copy = SG_MALLOC(sg_type, $input.len);
	sg_memcpy(copy, $input.array, $input.len);
	$1 = shogun::SGVector<sg_type>((SGTYPE *)copy, $input.len);
}

%typemap(out) shogun::SGVector<sg_type>
{
	$result.array = SG_MALLOC(sg_type, $1.vlen);
	$result.len = $1.vlen;
	$result.cap = $1.vlen;
	sg_memcpy($result.array, $1.vector, $1.vlen);
}
%enddef

/* Define concrete examples of the TYPEMAP_SG_VECTOR macros */
TYPEMAP_SGVECTOR(bool, bool)
TYPEMAP_SGVECTOR(char, int8)
TYPEMAP_SGVECTOR(byte, uint8_t)
TYPEMAP_SGVECTOR(int16, int16_t)
TYPEMAP_SGVECTOR(uint16, uint16_t)
TYPEMAP_SGVECTOR(int, int32_t)
TYPEMAP_SGVECTOR(uint, uint32_t)
TYPEMAP_SGVECTOR(int64, int64_t)
TYPEMAP_SGVECTOR(uint64, uint64_t)
TYPEMAP_SGVECTOR(float32, float32_t)
TYPEMAP_SGVECTOR(float64, float64_t)

#undef TYPEMAP_SGVECTOR

%define TYPEMAP_SGMATRIX(go_type, sg_type)
%typemap(gotype) shogun::SGMatrix<sg_type> %{[][]go_type%}

%typemap(in) shogun::SGMatrix<sg_type> {
	index_t cols = 0;
	for (auto i = 0; i < $input.len; i++)
	{
		_goslice_ *row = &((_goslice_ *)$input.array)[i];
		if (row->len > cols) cols = row->len;
	}

	sg_type* array = SG_MALLOC(sg_type, $input.len * cols);
	for (auto i = 0; i < $input.len; i++)
	{
		_goslice_ *row = &((_goslice_ *)$input.array)[i];
		sg_memcpy((void *)(array + (cols * i)), (sg_type *)row->array, row->len * sizeof(float));
	}
	$1 = shogun::SGMatrix<sg_type>((sg_type*)array, $input.len, cols, true);
}

%typemap(out) shogun::SGMatrix<sg_type> {
	sg_type* matrix = $1.matrix;
	auto rows = $1.num_rows;
	auto cols = $1.num_cols;

	_goslice_ *a;
	$result.array = SG_MALLOC($1.num_rows, sizeof(_goslice_));
	for (auto i = 0; i < rows; ++i)
	{
		_goslice_ *row = &((_goslice_ *)$result.array)[i];
		row->array = SG_MALLOC(sg_type, cols);
		row->len = cols;
		row->cap = cols;
		for (auto j = 0; j < cols; ++j)
		{
			((sg_type *)row->array)[j] = matrix[j+i*num_feat];
		}
	}
	$result.len = rows;
	$result.cap = $result.len;
}

%enddef

/* Define concrete examples of the TYPEMAP_SGMATRIX macros */
TYPEMAP_SGMATRIX(bool, bool)
TYPEMAP_SGMATRIX(char, int8)
TYPEMAP_SGMATRIX(byte, uint8_t)
TYPEMAP_SGMATRIX(int16, int16_t)
TYPEMAP_SGMATRIX(uint16, uint16_t)
TYPEMAP_SGMATRIX(int, int32_t)
TYPEMAP_SGMATRIX(uint, uint32_t)
TYPEMAP_SGMATRIX(int64, int64_t)
TYPEMAP_SGMATRIX(uint64, uint64_t)
TYPEMAP_SGMATRIX(float32, float32_t)
TYPEMAP_SGMATRIX(float64, float64_t)

#undef TYPEMAP_SGMATRIX
