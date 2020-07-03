/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Sergey Lisitsyn
 */

%{
#if ((OCTAVE_MAJOR_VERSION == 4) && (OCTAVE_MINOR_VERSION >= 4))
#include <octave/octave-config.h>
#else
#include <octave/config.h>
#endif

#include <octave/ov.h>
#include <octave/defun-dld.h>
#include <octave/error.h>
#include <octave/oct-obj.h>
#include <octave/pager.h>
#include <octave/symtab.h>
#include <octave/variables.h>
#include <octave/Cell.h>

#include <shogun/lib/DataType.h>

// this is for the hack that sets the number of threads to 1 below
// see #3772
#include <shogun/io/SGIO.h>
%}

/* One dimensional input arrays */
%define TYPEMAP_IN_SGVECTOR(oct_type_check, oct_type, oct_converter, sg_type, if_type, error_string)
%typemap(typecheck, precedence=SWIG_TYPECHECK_POINTER) shogun::SGVector<sg_type>
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
        /*error("Expected " error_string " Vector as argument");*/
        SWIG_fail;
    }

    m = mat_feat.oct_converter();

    void* copy=get_copy((void*) m.fortran_vec(), size_t(m.cols())*sizeof(sg_type));
    $1 = shogun::SGVector<sg_type>((sg_type*) copy, m.cols());
}
%typemap(freearg) shogun::SGVector<sg_type>
{
}
%enddef

/* Define concrete examples of the TYPEMAP_SG_VECTOR macros */
TYPEMAP_IN_SGVECTOR(is_bool_type, boolNDArray, bool_array_value, bool, bool, "Boolean")
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

    dim_vector vdims = dim_vector::alloc(2);
    vdims(0) = 1;
    vdims(1) = len;

    oct_type mat=oct_type(vdims);

    if (mat.cols() != len)
        SWIG_fail;

    for (int32_t i=0; i<len; i++)
        mat(i) = (if_type) vec[i];

    $result=mat;
}
%enddef

/* Define concrete examples of the TYPEMAP_OUT_SGVECTOR macros */
TYPEMAP_OUT_SGVECTOR(boolNDArray, bool, bool, "Boolean")
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
%typemap(typecheck, precedence=SWIG_TYPECHECK_POINTER) shogun::SGMatrix<sg_type>
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
        /*error("Expected " error_string " Matrix as argument");*/
        SWIG_fail;
    }

    m = mat_feat.oct_converter();

    void* copy=get_copy((void*) m.fortran_vec(), size_t(m.cols())*m.rows()*sizeof(sg_type));
    $1 = shogun::SGMatrix<sg_type>((sg_type*) copy, m.rows(), m.cols(), true);
}
%typemap(freearg) shogun::SGMatrix<sg_type>
{
}
%enddef

/* Define concrete examples of the TYPEMAP_IN_SGMATRIX macros */
TYPEMAP_IN_SGMATRIX(is_bool_type, boolNDArray, bool_array_value, bool, bool, "Boolean")
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

    dim_vector vdims = dim_vector::alloc(2);
    vdims(0) = num_feat;
    vdims(1) = num_vec;

    oct_type mat=oct_type(vdims);

    if (mat.rows() != num_feat || mat.cols() != num_vec)
        SWIG_fail;

    for (int32_t i=0; i<num_vec; i++)
    {
        for (int32_t j=0; j<num_feat; j++)
            mat(j,i) = (if_type) matrix[j+i*num_feat];
    }

    $result=mat;
}
%enddef

TYPEMAP_OUT_SGMATRIX(boolNDArray, bool, bool, "Boolean")
TYPEMAP_OUT_SGMATRIX(uint8NDArray, uint8_t, uint8_t, "Byte")
TYPEMAP_OUT_SGMATRIX(charMatrix, char, char, "Char")
TYPEMAP_OUT_SGMATRIX(int32NDArray, int32_t, int32_t, "Integer")
TYPEMAP_OUT_SGMATRIX(int16NDArray, int16_t, int16_t, "Short")
TYPEMAP_OUT_SGMATRIX(Matrix, float32_t, float32_t, "Single Precision")
TYPEMAP_OUT_SGMATRIX(Matrix, float64_t, float64_t, "Double Precision")
TYPEMAP_OUT_SGMATRIX(uint16NDArray, uint16_t, uint16_t, "Word")
#undef TYPEMAP_OUT_SGMATRIX


/* N-dimensional input arrays */
%define TYPEMAP_INND(oct_type_check, oct_type, oct_converter, sg_type, if_type, error_string)
%typemap(typecheck, precedence=SWIG_TYPECHECK_POINTER) shogun::SGNDArray<sg_type>
{
    const octave_value m=$input;
    $1 = (m.is_matrix_type() && m.oct_type_check()) ? 1 : 0;
}

%typemap(in) shogun::SGNDArray<sg_type>
{
    oct_type m;
    const octave_value mat_feat=$input;
    if (!mat_feat.is_matrix_type() || !mat_feat.oct_type_check())
    {
        /*error("Expected " error_string " Matrix as argument");*/
        SWIG_fail;
    }

    m = mat_feat.oct_converter();

    int32_t n = 1;
    index_t * sdims = SG_MALLOC(index_t, m.ndims());
    for (int32_t i = 0; i < m.ndims(); i++)
    {
        sdims[i] = m.dims().elem(i);
        n *= m.dims().elem(i);
    }

    void* copy=get_copy((void*) m.fortran_vec(), size_t(n*sizeof(sg_type)));
    $1 = shogun::SGNDArray<sg_type>((sg_type*) copy, sdims, m.ndims(), true);
}
%typemap(freearg) shogun::SGNDArray<sg_type>
{
}
%enddef

/* Define concrete examples of the TYPEMAP_INND macros */
TYPEMAP_INND(is_uint8_type, uint8NDArray, uint8_array_value, uint8_t, uint8_t, "Byte")
TYPEMAP_INND(is_char_matrix, charNDArray, char_matrix_value, char, char, "Char")
TYPEMAP_INND(is_int32_type, int32NDArray, int32_array_value, int32_t, int32_t, "Integer")
TYPEMAP_INND(is_int16_type, int16NDArray, int16_array_value, int16_t, int16_t, "Short")
TYPEMAP_INND(is_single_type, NDArray, array_value, float32_t, float32_t, "Single Precision")
TYPEMAP_INND(is_double_type, NDArray, array_value, float64_t, float64_t, "Double Precision")
TYPEMAP_INND(is_uint16_type, uint16NDArray, uint16_array_value, uint16_t, uint16_t, "Word")
#undef TYPEMAP_INND

/* N-dimensional output arrays */
%define TYPEMAP_OUTND(oct_type, sg_type, if_type, error_string)
%typemap(out) shogun::SGNDArray<sg_type>
{
    sg_type* array = $1.array;
    int32_t* dims = $1.dims;
    int32_t num_dims = $1.num_dims;

    dim_vector vdims = dim_vector::alloc(num_dims);

    int32_t n = 1;
    for (int32_t i = 0; i < num_dims; i++)
    {
        n *= dims[i];
        vdims(i) = (int32_t)dims[i];
    }

    oct_type mat = oct_type(vdims);

    for (int32_t i=0; i<n; i++)
        mat(i) = (if_type) array[i];

    $result=mat;
}
%enddef

TYPEMAP_OUTND(uint8NDArray, uint8_t, uint8_t, "Byte")
TYPEMAP_OUTND(charNDArray, char, char, "Char")
TYPEMAP_OUTND(int32NDArray, int32_t, int32_t, "Integer")
TYPEMAP_OUTND(int16NDArray, int16_t, int16_t, "Short")
TYPEMAP_OUTND(NDArray, float32_t, float32_t, "Single Precision")
TYPEMAP_OUTND(NDArray, float64_t, float64_t, "Double Precision")
TYPEMAP_OUTND(uint16NDArray, uint16_t, uint16_t, "Word")
#undef TYPEMAP_OUTND


/* input typemap for CStringFeatures<char> etc */
%define TYPEMAP_STRINGFEATURES_IN(oct_type_check, oct_type, oct_converter, sg_type, if_type, error_string)
%typemap(typecheck, precedence=SWIG_TYPECHECK_POINTER) std::vector<shogun::SGVector<sg_type>>, const std::vector<shogun::SGVector<sg_type>>&
{
    $1=0;
    octave_value arg=$input;
    if (arg.is_cell())
        $1=1;
    else if (arg.oct_type_check())
        $1=1;
}

%fragment(SWIG_AsVal_frag(std::vector<shogun::SGVector<sg_type>>), "header")
{
    int SWIG_AsVal_dec(std::vector<shogun::SGVector<sg_type>>)
        (const octave_value& arg, std::vector<shogun::SGVector<sg_type>>& strings)
    {
        if (arg.is_cell())
        {
            Cell c = arg.cell_value();
            int32_t num_strings=c.numel();
            ASSERT(num_strings>=1);
            strings.reserve(num_strings);

            for (int32_t i=0; i<num_strings; i++)
            {
                if (!c.elem(i).oct_type_check() || c.elem(i).rows()!=1)
                    return SWIG_ERROR;

                oct_type str=c.elem(i).oct_converter();

                int32_t len=str.cols();
                strings.emplace_back(len);

                for (int32_t j=0; j<len; j++)
                    strings.back().vector[j]=str(0,j);
            }
            return SWIG_OK;
        }
        else if (arg.oct_type_check())
        {
            oct_type data=arg.oct_converter();
            int32_t num_strings=data.cols();
            int32_t len=data.rows();
            strings.reserve(num_strings);
            ASSERT(num_strings>=1);

            for (int32_t i=0; i<num_strings; i++)
            {
                strings.emplace_back(len);

                for (int32_t j=0; j<len; j++)
                    strings.back().vector[j]=data(j,i);
            }
            return SWIG_OK;
        }
        else
        {
            return SWIG_ERROR;
        }
    }
}

%val_in_typemap(std::vector<shogun::SGVector<sg_type>>);
%enddef

TYPEMAP_STRINGFEATURES_IN(is_matrix_type() && arg.is_uint8_type, uint8NDArray, uint8_array_value, uint8_t, uint8_t, "Byte")
TYPEMAP_STRINGFEATURES_IN(is_char_matrix, charMatrix, char_matrix_value, char, char, "Char")
TYPEMAP_STRINGFEATURES_IN(is_matrix_type() && arg.is_int32_type, int32NDArray, int32_array_value, int32_t, int32_t, "Integer")
TYPEMAP_STRINGFEATURES_IN(is_matrix_type() && arg.is_int16_type, int16NDArray, int16_array_value, int16_t, int16_t, "Short")
TYPEMAP_STRINGFEATURES_IN(is_matrix_type() && arg.is_uint16_type, uint16NDArray, uint16_array_value, uint16_t, uint16_t, "Word")
#undef TYPEMAP_STRINGFEATURES_IN

/* output typemap for CStringFeatures */
%fragment(SWIG_From_frag(std::vector<shogun::SGVector<char>>), "header")
{
    Cell SWIG_From_dec(std::vector<shogun::SGVector<char>>)
        (const std::vector<shogun::SGVector<char>>& str)
    {
        int32_t num_strings = str.size();

        Cell c(num_strings, 1);

        for (int32_t i = 0; i < num_strings; i++) {
            std::stringstream ss;
            ss.write(str[i].vector, str[i].vlen);
            c(i)=ss.str();
        }

        return c;
    }
}
%val_out_typemap(std::vector<shogun::SGVector<char>>);

%define TYPEMAP_STRINGFEATURES_OUT(oct_type, sg_type)

/* output typemap for CStringFeatures */
%fragment(SWIG_From_frag(std::vector<shogun::SGVector<sg_type>>), "header")
{
    Cell SWIG_From_dec(std::vector<shogun::SGVector<sg_type>>)
        (const std::vector<shogun::SGVector<sg_type>>& strings)
    {
        int32_t num_strings = strings.size();

        Cell c(num_strings, 1);
        
        for (auto i : range(num_strings))
        {
            auto len = strings[i].vlen;
            dim_vector vdims = dim_vector::alloc(2);
            vdims(0) = 1;
            vdims(1) = len;
            auto vec=oct_type(vdims);
            
            for (auto j : range(len))
                vec(j) = strings[i].vector[j];

            c(i) = vec;
        }

        return c;
    }
}
%val_out_typemap(std::vector<shogun::SGVector<sg_type>>);
%enddef

TYPEMAP_STRINGFEATURES_OUT(uint16NDArray, uint16_t)
#undef TYPEMAP_STRINGFEATURES_OUT

/* input typemap for Sparse Features */
%define TYPEMAP_SPARSEFEATURES_IN(type,typecode)
%typemap(typecheck, precedence=SWIG_TYPECHECK_POINTER) shogun::SGSparseMatrix<type>
{
	const octave_value m=$input;

    $1 = (m.is_sparse_type() && m.is_double_type()) ? 1 : 0;
}
%typemap(in) shogun::SGSparseMatrix<type>
{
	const octave_value mat_feat = $input;
	if (!mat_feat.is_sparse_type() || !(mat_feat.is_double_type()))
	{
		SWIG_fail;
	}

	SparseMatrix sm = mat_feat.sparse_matrix_value ();
	int32_t num_vec=sm.cols();
	int32_t num_feat=sm.rows();
	int64_t nnz=sm.numel();

	SGSparseVector<type>* matrix = SG_MALLOC(SGSparseVector<type>, num_vec);
	for (int32_t i=0; i<num_vec; i++)
		new (&matrix[i]) SGSparseVector<type>();

	int64_t offset=0;
	for (int32_t i=0; i<num_vec; i++)
	{
		int32_t len=sm.cidx(i+1)-sm.cidx(i);
		matrix[i].num_feat_entries=len;

		if (len>0)
		{
			matrix[i].features=SG_MALLOC(SGSparseVectorEntry<type>, len);

			for (int32_t j=0; j<len; j++)
			{
				matrix[i].features[j].entry=sm.data(offset);
				matrix[i].features[j].feat_index=sm.ridx(offset);
				offset++;
			}
		}
		else
			matrix[i].features=NULL;
	}
	ASSERT(offset=nnz);
	$1 = shogun::SGSparseMatrix<type>(matrix, num_feat, num_vec, true);
}
%typemap(freearg) shogun::SGSparseMatrix<type>
{

}
%enddef
TYPEMAP_SPARSEFEATURES_IN(float64_t,     Matrix)
#undef TYPEMAP_SPARSEFEATURES_IN

/* output typemap for sparse features returns (data, row, ptr) */
%define TYPEMAP_SPARSEFEATURES_OUT(type,typecode)
%typemap(out) shogun::SGSparseMatrix<type>
{
	int32_t num_vec = $1.num_vectors;
	int32_t num_feat = $1.num_features;

	int64_t nnz = 0;
	for (int32_t i = 0; i < num_vec; i++)
	{
		int32_t len = $1.sparse_matrix[i].num_feat_entries;
		for (int32_t j = 0; j < len; j++)
		{
			nnz++;
		}
	}

	SparseMatrix sm((octave_idx_type) num_feat, (octave_idx_type) num_vec, (octave_idx_type) nnz);

	if(sm.cols() != num_vec || sm.rows() != num_feat)
	{
		SWIG_fail;
	}

	int64_t offset = 0;
	for (int32_t i = 0; i < num_vec; i++)
	{
		int32_t len = $1.sparse_matrix[i].num_feat_entries;
		sm.cidx(i) = offset;
		for (int32_t j = 0; j < len; j++)
		{
			sm.data(offset) = $1.sparse_matrix[i].features[j].entry;
			sm.ridx(offset) = $1.sparse_matrix[i].features[j].feat_index;
			offset++;
		}
	}
	sm.cidx(num_vec) = offset;
	ASSERT(offset=nnz);

	$result = sm;
}
%enddef

TYPEMAP_SPARSEFEATURES_OUT(float64_t,     NPY_FLOAT64)
#undef TYPEMAP_SPARSEFEATURES_OUT

%init %{
	// set number of threads to 1
	// see issue #3772
	io::warn("Using Shogun single-threaded. Multi-threaded Octave is currently broken. See https://github.com/shogun-toolbox/shogun/issues/3772");
	shogun::env()->set_num_threads(1);
%}
