/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2011-2013 Heiko Strathmann
 * Written (W) 2012 Fernando Jose Iglesias Garcia
 * Written (W) 2010,2012 Soeren Sonnenburg
 * Copyright (C) 2010 Berlin Institute of Technology
 * Copyright (C) 2012 Soeren Sonnenburg
 */

#include <shogun/lib/config.h>
#include <shogun/io/SGIO.h>
#include <shogun/io/File.h>
#include <shogun/lib/SGMatrix.h>
#include <shogun/lib/SGVector.h>
#include <shogun/mathematics/Math.h>
#include <shogun/mathematics/lapack.h>
#include <limits>

#ifdef HAVE_EIGEN3
#include <shogun/mathematics/eigen3.h>
#endif

namespace shogun {

template <class T>
SGMatrix<T>::SGMatrix() : SGReferencedData()
{
	init_data();
}

template <class T>
SGMatrix<T>::SGMatrix(bool ref_counting) : SGReferencedData(ref_counting)
{
	init_data();
}

template <class T>
SGMatrix<T>::SGMatrix(T* m, index_t nrows, index_t ncols, bool ref_counting)
	: SGReferencedData(ref_counting), matrix(m),
	num_rows(nrows), num_cols(ncols) { }

template <class T>
SGMatrix<T>::SGMatrix(index_t nrows, index_t ncols, bool ref_counting)
	: SGReferencedData(ref_counting), num_rows(nrows), num_cols(ncols)
{
	matrix=SG_MALLOC(T, ((int64_t) nrows)*ncols);
}

template <class T>
SGMatrix<T>::SGMatrix(SGVector<T> vec) : SGReferencedData(vec)
{
	REQUIRE(vec.vector, "Vector not initialized!\n");
	matrix=vec.vector;
	num_rows=vec.vlen;
	num_cols=1;
}

template <class T>
SGMatrix<T>::SGMatrix(SGVector<T> vec, index_t nrows, index_t ncols)
: SGReferencedData(vec)
{
	REQUIRE(vec.vector, "Vector not initialized!\n");
	REQUIRE(nrows>0, "Number of rows (%d) has to be a positive integer!\n", nrows);
	REQUIRE(ncols>0, "Number of cols (%d) has to be a positive integer!\n", ncols);
	REQUIRE(vec.vlen==nrows*ncols, "Number of elements in the matrix (%d) must "
			"be the same as the number of elements in the vector (%d)!\n",
			nrows*ncols, vec.vlen);
	matrix=vec.vector;
	num_rows=nrows;
	num_cols=ncols;
}

template <class T>
SGMatrix<T>::SGMatrix(const SGMatrix &orig) : SGReferencedData(orig)
{
	copy_data(orig);
}

#ifdef HAVE_EIGEN3
template <class T>
SGMatrix<T>::SGMatrix(EigenMatrixXt& mat)
: SGReferencedData(false), matrix(mat.data()),
	num_rows(mat.rows()), num_cols(mat.cols())
{

}

template <class T>
SGMatrix<T>::operator EigenMatrixXtMap() const
{
	return EigenMatrixXtMap(matrix, num_rows, num_cols);
}
#endif

template <class T>
SGMatrix<T>::~SGMatrix()
{
	unref();
}

template <class T>
bool SGMatrix<T>::operator==(SGMatrix<T>& other)
{
	if (num_rows!=other.num_rows || num_cols!=other.num_cols)
		return false;

	if (matrix!=other.matrix)
		return false;

	return true;
}

template <class T>
bool SGMatrix<T>::equals(SGMatrix<T>& other)
{
	if (num_rows!=other.num_rows || num_cols!=other.num_cols)
		return false;

	for (int64_t i=0; i<int64_t(num_rows)*num_cols; ++i)
	{
		if (matrix[i]!=other.matrix[i])
			return false;
	}

	return true;
}

template <class T>
void SGMatrix<T>::set_const(T const_elem)
{
	for (int64_t i=0; i<int64_t(num_rows)*num_cols; i++)
		matrix[i]=const_elem ;
}

template <class T>
void SGMatrix<T>::zero()
{
	if (matrix && (int64_t(num_rows)*num_cols))
		set_const(0);
}

template <>
void SGMatrix<complex128_t>::zero()
{
	if (matrix && (int64_t(num_rows)*num_cols))
		set_const(complex128_t(0.0));
}

template <class T>
bool SGMatrix<T>::is_symmetric()
{
	if (num_rows!=num_cols)
		return false;
	for (int i=0; i<num_rows; ++i)
	{
		for (int j=i+1; j<num_cols; ++j)
		{
			if (matrix[j*num_rows+i]!=matrix[i*num_rows+j])
				return false;
		}
	}
	return true;
}

template <>
bool SGMatrix<float32_t>::is_symmetric()
{
	if (num_rows!=num_cols)
		return false;
	for (int i=0; i<num_rows; ++i)
	{
		for (int j=i+1; j<num_cols; ++j)
		{
			if (!CMath::fequals<float32_t>(matrix[j*num_rows+i],
						matrix[i*num_rows+j], FLT_EPSILON))
				return false;
		}
	}
	return true;
}

template <>
bool SGMatrix<float64_t>::is_symmetric()
{
	if (num_rows!=num_cols)
		return false;
	for (int i=0; i<num_rows; ++i)
	{
		for (int j=i+1; j<num_cols; ++j)
		{
			if (!CMath::fequals<float64_t>(matrix[j*num_rows+i],
						matrix[i*num_rows+j], DBL_EPSILON))
				return false;
		}
	}
	return true;
}

template <>
bool SGMatrix<floatmax_t>::is_symmetric()
{
	if (num_rows!=num_cols)
		return false;
	for (int i=0; i<num_rows; ++i)
	{
		for (int j=i+1; j<num_cols; ++j)
		{
			if (!CMath::fequals<floatmax_t>(matrix[j*num_rows+i],
						matrix[i*num_rows+j], LDBL_EPSILON))
				return false;
		}
	}
	return true;
}

template <>
bool SGMatrix<complex128_t>::is_symmetric()
{
	if (num_rows!=num_cols)
		return false;
	for (int i=0; i<num_rows; ++i)
	{
		for (int j=i+1; j<num_cols; ++j)
		{
			if (!(CMath::fequals<float64_t>(matrix[j*num_rows+i].real(),
						matrix[i*num_rows+j].real(), DBL_EPSILON) &&
					CMath::fequals<float64_t>(matrix[j*num_rows+i].imag(),
						matrix[i*num_rows+j].imag(), DBL_EPSILON)))
				return false;
		}
	}
	return true;
}

template <class T>
T SGMatrix<T>::max_single()
{
	T max=matrix[0];
	for (int64_t i=1; i<int64_t(num_rows)*num_cols; ++i)
	{
		if (matrix[i]>max)
			max=matrix[i];
	}

	return max;
}

template <>
complex128_t SGMatrix<complex128_t>::max_single()
{
	SG_SERROR("SGMatrix::max_single():: Not supported for complex128_t\n");
	return complex128_t(0.0);
}

template <class T>
SGMatrix<T> SGMatrix<T>::clone()
{
	return SGMatrix<T>(clone_matrix(matrix, num_rows, num_cols),
			num_rows, num_cols);
}

template <class T>
T* SGMatrix<T>::clone_matrix(const T* matrix, int32_t nrows, int32_t ncols)
{
	T* result = SG_MALLOC(T, int64_t(nrows)*ncols);
	for (int64_t i=0; i<int64_t(nrows)*ncols; i++)
		result[i]=matrix[i];

	return result;
}

template <class T>
void SGMatrix<T>::transpose_matrix(
	T*& matrix, int32_t& num_feat, int32_t& num_vec)
{
	/* this should be done in-place! Heiko */
	T* transposed=SG_MALLOC(T, int64_t(num_vec)*num_feat);
	for (int64_t i=0; i<num_vec; i++)
	{
		for (int64_t j=0; j<num_feat; j++)
			transposed[i+j*num_vec]=matrix[i*num_feat+j];
	}

	SG_FREE(matrix);
	matrix=transposed;

	CMath::swap(num_feat, num_vec);
}

template <class T>
void SGMatrix<T>::create_diagonal_matrix(T* matrix, T* v,int32_t size)
{
	for(int64_t i=0;i<size;i++)
	{
		for(int64_t j=0;j<size;j++)
		{
			if(i==j)
				matrix[j*size+i]=v[i];
			else
				matrix[j*size+i]=0;
		}
	}
}

template <class T>
float64_t SGMatrix<T>::trace(
	float64_t* mat, int32_t cols, int32_t rows)
{
	float64_t trace=0;
	for (int64_t i=0; i<rows; i++)
		trace+=mat[i*cols+i];
	return trace;
}

template <class T>
T* SGMatrix<T>::get_row_sum(T* matrix, int32_t m, int32_t n)
{
	T* rowsums=SG_CALLOC(T, n);

	for (int64_t i=0; i<n; i++)
	{
		for (int64_t j=0; j<m; j++)
			rowsums[i]+=matrix[j+i*m];
	}
	return rowsums;
}

template <class T>
T* SGMatrix<T>::get_column_sum(T* matrix, int32_t m, int32_t n)
{
	T* colsums=SG_CALLOC(T, m);

	for (int64_t i=0; i<n; i++)
	{
		for (int64_t j=0; j<m; j++)
			colsums[j]+=matrix[j+i*m];
	}
	return colsums;
}

template <class T>
void SGMatrix<T>::center()
{
	center_matrix(matrix, num_rows, num_cols);
}

template <class T>
void SGMatrix<T>::center_matrix(T* matrix, int32_t m, int32_t n)
{
	float64_t num_data=n;

	T* colsums=get_column_sum(matrix, m,n);
	T* rowsums=get_row_sum(matrix, m,n);

	for (int32_t i=0; i<m; i++)
		colsums[i]/=num_data;
	for (int32_t j=0; j<n; j++)
		rowsums[j]/=num_data;

	T s=SGVector<T>::sum(rowsums, n)/num_data;

	for (int64_t i=0; i<n; i++)
	{
		for (int64_t j=0; j<m; j++)
			matrix[i*m+j]+=s-colsums[j]-rowsums[i];
	}

	SG_FREE(rowsums);
	SG_FREE(colsums);
}

template <class T>
void SGMatrix<T>::remove_column_mean()
{
	/* compute "row" sums (which I would call column sums), i.e. sum of all
	 * elements in a fixed column */
	T* means=get_row_sum(matrix, num_rows, num_cols);

	/* substract column mean from every corresponding entry */
	for (int64_t i=0; i<num_cols; ++i)
	{
		means[i]/=num_rows;
		for (int64_t j=0; j<num_rows; ++j)
			matrix[i*num_rows+j]-=means[i];
	}

	SG_FREE(means);
}

template<class T> void SGMatrix<T>::display_matrix(const char* name) const
{
	display_matrix(matrix, num_rows, num_cols, name);
}

template <class T>
void SGMatrix<T>::display_matrix(
	const SGMatrix<T> matrix, const char* name,
	const char* prefix)
{
	matrix.display_matrix();
}

template <>
void SGMatrix<bool>::display_matrix(
	const bool* matrix, int32_t rows, int32_t cols, const char* name,
	const char* prefix)
{
	ASSERT(rows>=0 && cols>=0)
	SG_SPRINT("%s%s=[\n", prefix, name)
	for (int64_t i=0; i<rows; i++)
	{
		SG_SPRINT("%s[", prefix)
		for (int64_t j=0; j<cols; j++)
			SG_SPRINT("%s\t%d%s", prefix, matrix[j*rows+i] ? 1 : 0,
				j==cols-1? "" : ",");
		SG_SPRINT("%s]%s\n", prefix, i==rows-1? "" : ",")
	}
	SG_SPRINT("%s]\n", prefix)
}

template <>
void SGMatrix<char>::display_matrix(
	const char* matrix, int32_t rows, int32_t cols, const char* name,
	const char* prefix)
{
	ASSERT(rows>=0 && cols>=0)
	SG_SPRINT("%s%s=[\n", prefix, name)
	for (int64_t i=0; i<rows; i++)
	{
		SG_SPRINT("%s[", prefix)
		for (int64_t j=0; j<cols; j++)
			SG_SPRINT("%s\t%c%s", prefix, matrix[j*rows+i],
				j==cols-1? "" : ",");
		SG_SPRINT("%s]%s\n", prefix, i==rows-1? "" : ",")
	}
	SG_SPRINT("%s]\n", prefix)
}

template <>
void SGMatrix<int8_t>::display_matrix(
	const int8_t* matrix, int32_t rows, int32_t cols, const char* name,
	const char* prefix)
{
	ASSERT(rows>=0 && cols>=0)
	SG_SPRINT("%s%s=[\n", prefix, name)
	for (int64_t i=0; i<rows; i++)
	{
		SG_SPRINT("%s[", prefix)
		for (int64_t j=0; j<cols; j++)
			SG_SPRINT("%s\t%d%s", prefix, matrix[j*rows+i],
				j==cols-1? "" : ",");
		SG_SPRINT("%s]%s\n", prefix, i==rows-1? "" : ",")
	}
	SG_SPRINT("%s]\n", prefix)
}

template <>
void SGMatrix<uint8_t>::display_matrix(
	const uint8_t* matrix, int32_t rows, int32_t cols, const char* name,
	const char* prefix)
{
	ASSERT(rows>=0 && cols>=0)
	SG_SPRINT("%s%s=[\n", prefix, name)
	for (int64_t i=0; i<rows; i++)
	{
		SG_SPRINT("%s[", prefix)
		for (int64_t j=0; j<cols; j++)
			SG_SPRINT("%s\t%d%s", prefix, matrix[j*rows+i],
				j==cols-1? "" : ",");
		SG_SPRINT("%s]%s\n", prefix, i==rows-1? "" : ",")
	}
	SG_SPRINT("%s]\n", prefix)
}

template <>
void SGMatrix<int16_t>::display_matrix(
	const int16_t* matrix, int32_t rows, int32_t cols, const char* name,
	const char* prefix)
{
	ASSERT(rows>=0 && cols>=0)
	SG_SPRINT("%s%s=[\n", prefix, name)
	for (int64_t i=0; i<rows; i++)
	{
		SG_SPRINT("%s[", prefix)
		for (int64_t j=0; j<cols; j++)
			SG_SPRINT("%s\t%d%s", prefix, matrix[j*rows+i],
				j==cols-1? "" : ",");
		SG_SPRINT("%s]%s\n", prefix, i==rows-1? "" : ",")
	}
	SG_SPRINT("%s]\n", prefix)
}

template <>
void SGMatrix<uint16_t>::display_matrix(
	const uint16_t* matrix, int32_t rows, int32_t cols, const char* name,
	const char* prefix)
{
	ASSERT(rows>=0 && cols>=0)
	SG_SPRINT("%s%s=[\n", prefix, name)
	for (int64_t i=0; i<rows; i++)
	{
		SG_SPRINT("%s[", prefix)
		for (int64_t j=0; j<cols; j++)
			SG_SPRINT("%s\t%d%s", prefix, matrix[j*rows+i],
				j==cols-1? "" : ",");
		SG_SPRINT("%s]%s\n", prefix, i==rows-1? "" : ",")
	}
	SG_SPRINT("%s]\n", prefix)
}


template <>
void SGMatrix<int32_t>::display_matrix(
	const int32_t* matrix, int32_t rows, int32_t cols, const char* name,
	const char* prefix)
{
	ASSERT(rows>=0 && cols>=0)
	SG_SPRINT("%s%s=[\n", prefix, name)
	for (int64_t i=0; i<rows; i++)
	{
		SG_SPRINT("%s[", prefix)
		for (int64_t j=0; j<cols; j++)
			SG_SPRINT("%s\t%d%s", prefix, matrix[j*rows+i],
				j==cols-1? "" : ",");
		SG_SPRINT("%s]%s\n", prefix, i==rows-1? "" : ",")
	}
	SG_SPRINT("%s]\n", prefix)
}

template <>
void SGMatrix<uint32_t>::display_matrix(
	const uint32_t* matrix, int32_t rows, int32_t cols, const char* name,
	const char* prefix)
{
	ASSERT(rows>=0 && cols>=0)
	SG_SPRINT("%s%s=[\n", prefix, name)
	for (int64_t i=0; i<rows; i++)
	{
		SG_SPRINT("%s[", prefix)
		for (int64_t j=0; j<cols; j++)
			SG_SPRINT("%s\t%d%s", prefix, matrix[j*rows+i],
				j==cols-1? "" : ",");
		SG_SPRINT("%s]%s\n", prefix, i==rows-1? "" : ",")
	}
	SG_SPRINT("%s]\n", prefix)
}
template <>
void SGMatrix<int64_t>::display_matrix(
	const int64_t* matrix, int32_t rows, int32_t cols, const char* name,
	const char* prefix)
{
	ASSERT(rows>=0 && cols>=0)
	SG_SPRINT("%s%s=[\n", prefix, name)
	for (int64_t i=0; i<rows; i++)
	{
		SG_SPRINT("%s[", prefix)
		for (int64_t j=0; j<cols; j++)
			SG_SPRINT("%s\t%d%s", prefix, matrix[j*rows+i],
				j==cols-1? "" : ",");
		SG_SPRINT("%s]%s\n", prefix, i==rows-1? "" : ",")
	}
	SG_SPRINT("%s]\n", prefix)
}

template <>
void SGMatrix<uint64_t>::display_matrix(
	const uint64_t* matrix, int32_t rows, int32_t cols, const char* name,
	const char* prefix)
{
	ASSERT(rows>=0 && cols>=0)
	SG_SPRINT("%s%s=[\n", prefix, name)
	for (int64_t i=0; i<rows; i++)
	{
		SG_SPRINT("%s[", prefix)
		for (int64_t j=0; j<cols; j++)
			SG_SPRINT("%s\t%d%s", prefix, matrix[j*rows+i],
				j==cols-1? "" : ",");
		SG_SPRINT("%s]%s\n", prefix, i==rows-1? "" : ",")
	}
	SG_SPRINT("%s]\n", prefix)
}

template <>
void SGMatrix<float32_t>::display_matrix(
	const float32_t* matrix, int32_t rows, int32_t cols, const char* name,
	const char* prefix)
{
	ASSERT(rows>=0 && cols>=0)
	SG_SPRINT("%s%s=[\n", prefix, name)
	for (int64_t i=0; i<rows; i++)
	{
		SG_SPRINT("%s[", prefix)
		for (int64_t j=0; j<cols; j++)
			SG_SPRINT("%s\t%.18g%s", prefix, (float) matrix[j*rows+i],
				j==cols-1? "" : ",");
		SG_SPRINT("%s]%s\n", prefix, i==rows-1? "" : ",")
	}
	SG_SPRINT("%s]\n", prefix)
}

template <>
void SGMatrix<float64_t>::display_matrix(
	const float64_t* matrix, int32_t rows, int32_t cols, const char* name,
	const char* prefix)
{
	ASSERT(rows>=0 && cols>=0)
	SG_SPRINT("%s%s=[\n", prefix, name)
	for (int64_t i=0; i<rows; i++)
	{
		SG_SPRINT("%s[", prefix)
		for (int64_t j=0; j<cols; j++)
			SG_SPRINT("%s\t%.18g%s", prefix, (double) matrix[j*rows+i],
				j==cols-1? "" : ",");
		SG_SPRINT("%s]%s\n", prefix, i==rows-1? "" : ",")
	}
	SG_SPRINT("%s]\n", prefix)
}

template <>
void SGMatrix<floatmax_t>::display_matrix(
	const floatmax_t* matrix, int32_t rows, int32_t cols, const char* name,
	const char* prefix)
{
	ASSERT(rows>=0 && cols>=0)
	SG_SPRINT("%s%s=[\n", prefix, name)
	for (int64_t i=0; i<rows; i++)
	{
		SG_SPRINT("%s[", prefix)
		for (int64_t j=0; j<cols; j++)
			SG_SPRINT("%s\t%.18g%s", prefix, (double) matrix[j*rows+i],
				j==cols-1? "" : ",");
		SG_SPRINT("%s]%s\n", prefix, i==rows-1? "" : ",")
	}
	SG_SPRINT("%s]\n", prefix)
}

template <>
void SGMatrix<complex128_t>::display_matrix(
	const complex128_t* matrix, int32_t rows, int32_t cols, const char* name,
	const char* prefix)
{
	ASSERT(rows>=0 && cols>=0)
	SG_SPRINT("%s%s=[\n", prefix, name)
	for (int64_t i=0; i<rows; i++)
	{
		SG_SPRINT("%s[", prefix)
		for (int64_t j=0; j<cols; j++)
			SG_SPRINT("%s\t(%.18g+i%.18g)%s", prefix, matrix[j*rows+i].real(),
				matrix[j*rows+i].imag(), j==cols-1? "" : ",");
		SG_SPRINT("%s]%s\n", prefix, i==rows-1? "" : ",")
	}
	SG_SPRINT("%s]\n", prefix)
}

template <>
SGMatrix<char> SGMatrix<char>::create_identity_matrix(index_t size, char scale)
{
	SG_SNOTIMPLEMENTED
	return SGMatrix<char>();
}

template <>
SGMatrix<int8_t> SGMatrix<int8_t>::create_identity_matrix(index_t size, int8_t scale)
{
	SGMatrix<int8_t> I(size, size);
	for (index_t i=0; i<size; ++i)
	{
		for (index_t j=0; j<size; ++j)
			I(i,j)=i==j ? scale : 0.0;
	}

	return I;
}

template <>
SGMatrix<uint8_t> SGMatrix<uint8_t>::create_identity_matrix(index_t size, uint8_t scale)
{
	SGMatrix<uint8_t> I(size, size);
	for (index_t i=0; i<size; ++i)
	{
		for (index_t j=0; j<size; ++j)
			I(i,j)=i==j ? scale : 0.0;
	}

	return I;
}

template <>
SGMatrix<bool> SGMatrix<bool>::create_identity_matrix(index_t size, bool scale)
{
	SGMatrix<bool> I(size, size);
	for (index_t i=0; i<size; ++i)
	{
		for (index_t j=0; j<size; ++j)
			I(i,j)=i==j ? scale : (!scale);
	}

	return I;
}

template <>
SGMatrix<int16_t> SGMatrix<int16_t>::create_identity_matrix(index_t size, int16_t scale)
{
	SGMatrix<int16_t> I(size, size);
	for (index_t i=0; i<size; ++i)
	{
		for (index_t j=0; j<size; ++j)
			I(i,j)=i==j ? scale : 0.0;
	}

	return I;
}

template <>
SGMatrix<uint16_t> SGMatrix<uint16_t>::create_identity_matrix(index_t size, uint16_t scale)
{
	SGMatrix<uint16_t> I(size, size);
	for (index_t i=0; i<size; ++i)
	{
		for (index_t j=0; j<size; ++j)
			I(i,j)=i==j ? scale : 0.0;
	}

	return I;
}

template <>
SGMatrix<int32_t> SGMatrix<int32_t>::create_identity_matrix(index_t size, int32_t scale)
{
	SGMatrix<int32_t> I(size, size);
	for (index_t i=0; i<size; ++i)
	{
		for (index_t j=0; j<size; ++j)
			I(i,j)=i==j ? scale : 0.0;
	}

	return I;
}

template <>
SGMatrix<uint32_t> SGMatrix<uint32_t>::create_identity_matrix(index_t size, uint32_t scale)
{
	SGMatrix<uint32_t> I(size, size);
	for (index_t i=0; i<size; ++i)
	{
		for (index_t j=0; j<size; ++j)
			I(i,j)=i==j ? scale : 0.0;
	}

	return I;
}

template <>
SGMatrix<int64_t> SGMatrix<int64_t>::create_identity_matrix(index_t size, int64_t scale)
{
	SGMatrix<int64_t> I(size, size);
	for (index_t i=0; i<size; ++i)
	{
		for (index_t j=0; j<size; ++j)
			I(i,j)=i==j ? scale : 0.0;
	}

	return I;
}

template <>
SGMatrix<uint64_t> SGMatrix<uint64_t>::create_identity_matrix(index_t size, uint64_t scale)
{
	SGMatrix<uint64_t> I(size, size);
	for (index_t i=0; i<size; ++i)
	{
		for (index_t j=0; j<size; ++j)
			I(i,j)=i==j ? scale : 0.0;
	}

	return I;
}

template <>
SGMatrix<float32_t> SGMatrix<float32_t>::create_identity_matrix(index_t size, float32_t scale)
{
	SGMatrix<float32_t> I(size, size);
	for (index_t i=0; i<size; ++i)
	{
		for (index_t j=0; j<size; ++j)
			I(i,j)=i==j ? scale : 0.0;
	}

	return I;
}

template <>
SGMatrix<float64_t> SGMatrix<float64_t>::create_identity_matrix(index_t size, float64_t scale)
{
	SGMatrix<float64_t> I(size, size);
	for (index_t i=0; i<size; ++i)
	{
		for (index_t j=0; j<size; ++j)
			I(i,j)=i==j ? scale : 0.0;
	}

	return I;
}

template <>
SGMatrix<floatmax_t> SGMatrix<floatmax_t>::create_identity_matrix(index_t size, floatmax_t scale)
{
	SGMatrix<floatmax_t> I(size, size);
	for (index_t i=0; i<size; ++i)
	{
		for (index_t j=0; j<size; ++j)
			I(i,j)=i==j ? scale : 0.0;
	}

	return I;
}

template <>
SGMatrix<complex128_t> SGMatrix<complex128_t>::create_identity_matrix(index_t size, complex128_t scale)
{
	SGMatrix<complex128_t> I(size, size);
	for (index_t i=0; i<size; ++i)
	{
		for (index_t j=0; j<size; ++j)
			I(i,j)=i==j ? scale : complex128_t(0.0);
	}

	return I;
}

//Howto construct the pseudo inverse (from "The Matrix Cookbook")
//
//Assume A does not have full rank, i.e. A is n \times m and rank(A) = r < min(n;m).
//
//The matrix A+ known as the pseudo inverse is unique and does always exist.
//
//The pseudo inverse A+ can be constructed from the singular value
//decomposition A = UDV^T , by  A^+ = V(D+)U^T.

#ifdef HAVE_LAPACK
template <class T>
float64_t* SGMatrix<T>::pinv(
		float64_t* matrix, int32_t rows, int32_t cols, float64_t* target)
{
	if (!target)
		target=SG_MALLOC(float64_t, rows*cols);

	char jobu='A';
	char jobvt='A';
	int m=rows; /* for calling external lib */
	int n=cols; /* for calling external lib */
	int lda=m; /* for calling external lib */
	int ldu=m; /* for calling external lib */
	int ldvt=n; /* for calling external lib */
	int info=-1; /* for calling external lib */
	int32_t lsize=CMath::min((int32_t) m, (int32_t) n);
	double* s=SG_MALLOC(double, lsize);
	double* u=SG_MALLOC(double, m*m);
	double* vt=SG_MALLOC(double, n*n);

	wrap_dgesvd(jobu, jobvt, m, n, matrix, lda, s, u, ldu, vt, ldvt, &info);
	ASSERT(info==0)

	for (int64_t i=0; i<n; i++)
	{
		for (int64_t j=0; j<lsize; j++)
			vt[i*n+j]=vt[i*n+j]/s[j];
	}

	cblas_dgemm(CblasColMajor, CblasTrans, CblasTrans, m, n, m, 1.0, vt, ldvt, u, ldu, 0, target, m);

	SG_FREE(u);
	SG_FREE(vt);
	SG_FREE(s);

	return target;
}

/// inverses square matrix in-place
template <class T>
void SGMatrix<T>::inverse(SGMatrix<float64_t> matrix)
{
	ASSERT(matrix.num_cols==matrix.num_rows)
	int32_t* ipiv = SG_MALLOC(int32_t, matrix.num_cols);
	clapack_dgetrf(CblasColMajor,matrix.num_cols,matrix.num_cols,matrix.matrix,matrix.num_cols,ipiv);
	clapack_dgetri(CblasColMajor,matrix.num_cols,matrix.matrix,matrix.num_cols,ipiv);
	SG_FREE(ipiv);
}

template <class T>
SGVector<float64_t> SGMatrix<T>::compute_eigenvectors(SGMatrix<float64_t> matrix)
{
	if (matrix.num_rows!=matrix.num_cols)
	{
		SG_SERROR("SGMatrix::compute_eigenvectors(SGMatrix<float64_t>): matrix"
				" rows and columns are not equal!\n");
	}

	/* use reference counting for SGVector */
	SGVector<float64_t> result(NULL, 0, true);
	result.vlen=matrix.num_rows;
	result.vector=compute_eigenvectors(matrix.matrix, matrix.num_rows,
			matrix.num_rows);
	return result;
}

template <class T>
double* SGMatrix<T>::compute_eigenvectors(double* matrix, int n, int m)
{
	ASSERT(n == m)

	char V='V';
	char U='U';
	int info;
	int ord=n;
	int lda=n;
	double* eigenvalues=SG_CALLOC(float64_t, n+1);

	// lapack sym matrix eigenvalues+vectors
	wrap_dsyev(V, U,  ord, matrix, lda,
			eigenvalues, &info);

	if (info!=0)
		SG_SERROR("DSYEV failed with code %d\n", info)

	return eigenvalues;
}

template <class T>
void SGMatrix<T>::compute_few_eigenvectors(double* matrix_, double*& eigenvalues, double*& eigenvectors,
                                           int n, int il, int iu)
{
	eigenvalues = SG_MALLOC(double, n);
	eigenvectors = SG_MALLOC(double, (iu-il+1)*n);
	int status = 0;
	wrap_dsyevr('V','U',n,matrix_,n,il,iu,eigenvalues,eigenvectors,&status);
	ASSERT(status==0)
}

#endif //HAVE_LAPACK

template <class T>
SGMatrix<float64_t> SGMatrix<T>::matrix_multiply(
		SGMatrix<float64_t> A, SGMatrix<float64_t> B,
		bool transpose_A, bool transpose_B, float64_t scale)
{
	/* these variables store size of transposed matrices*/
	index_t cols_A=transpose_A ? A.num_rows : A.num_cols;
	index_t rows_A=transpose_A ? A.num_cols : A.num_rows;
	index_t rows_B=transpose_B ? B.num_cols : B.num_rows;
	index_t cols_B=transpose_B ? B.num_rows : B.num_cols;

	/* do a dimension check */
	if (cols_A!=rows_B)
	{
		SG_SERROR("SGMatrix::matrix_multiply(): Dimension mismatch: "
				"A(%dx%d)*B(%dx%D)\n", rows_A, cols_A, rows_B, cols_B);
	}

	/* allocate result matrix */
	SGMatrix<float64_t> C(rows_A, cols_B);
	C.zero();
#ifdef HAVE_LAPACK
	/* multiply */
	cblas_dgemm(CblasColMajor,
			transpose_A ? CblasTrans : CblasNoTrans,
			transpose_B ? CblasTrans : CblasNoTrans,
			rows_A, cols_B, cols_A, scale,
			A.matrix, A.num_rows, B.matrix, B.num_rows,
			0.0, C.matrix, C.num_rows);
#else
	/* C(i,j) = scale * \Sigma A(i,k)*B(k,j) */
	for (int32_t i=0; i<rows_A; i++)
	{
		for (int32_t j=0; j<cols_B; j++)
		{
			for (int32_t k=0; k<cols_A; k++)
			{
				float64_t x1=transpose_A ? A(k,i):A(i,k);
				float64_t x2=transpose_B ? B(j,k):B(k,j);
				C(i,j)+=x1*x2;
			}

			C(i,j)*=scale;
		}
	}
#endif //HAVE_LAPACK

	return C;
}

template<class T>
SGMatrix<T> SGMatrix<T>::get_allocated_matrix(index_t num_rows,
		index_t num_cols, SGMatrix<T> pre_allocated)
{
	SGMatrix<T> result;

	/* evtl use pre-allocated space */
	if (pre_allocated.matrix)
	{
		result=pre_allocated;

		/* check dimension consistency */
		if (pre_allocated.num_rows!=num_rows ||
				pre_allocated.num_cols!=num_cols)
		{
			SG_SERROR("SGMatrix<T>::get_allocated_matrix(). Provided target"
					"matrix dimensions (%dx%d) do not match passed data "
					"dimensions (%dx%d)!\n", pre_allocated.num_rows,
					pre_allocated.num_cols, num_rows, num_cols);
		}
	}
	else
	{
		/* otherwise, allocate space */
		result=SGMatrix<T>(num_rows, num_cols);
	}

	return result;
}

template<class T>
void SGMatrix<T>::copy_data(const SGReferencedData &orig)
{
	matrix=((SGMatrix*)(&orig))->matrix;
	num_rows=((SGMatrix*)(&orig))->num_rows;
	num_cols=((SGMatrix*)(&orig))->num_cols;
}

template<class T>
void SGMatrix<T>::init_data()
{
	matrix=NULL;
	num_rows=0;
	num_cols=0;
}

template<class T>
void SGMatrix<T>::free_data()
{
	SG_FREE(matrix);
	matrix=NULL;
	num_rows=0;
	num_cols=0;
}

template<class T>
void SGMatrix<T>::load(CFile* loader)
{
	ASSERT(loader)
	unref();

	SG_SET_LOCALE_C;
	SGMatrix<T> mat;
	loader->get_matrix(mat.matrix, mat.num_rows, mat.num_cols);
	copy_data(mat);
	copy_refcount(mat);
	ref();
	SG_RESET_LOCALE;
}

template<>
void SGMatrix<complex128_t>::load(CFile* loader)
{
	SG_SERROR("SGMatrix::load():: Not supported for complex128_t\n");
}

template<class T>
void SGMatrix<T>::save(CFile* writer)
{
	ASSERT(writer)
	SG_SET_LOCALE_C;
	writer->set_matrix(matrix, num_rows, num_cols);
	SG_RESET_LOCALE;
}

template<>
void SGMatrix<complex128_t>::save(CFile* saver)
{
	SG_SERROR("SGMatrix::save():: Not supported for complex128_t\n");
}

template<class T>
SGVector<T> SGMatrix<T>::get_row_vector(index_t row) const
{
	SGVector<T> rowv(num_cols);
	for (int64_t i = 0; i < num_cols; i++)
	{
		rowv[i] = matrix[i*num_rows+row];
	}
	return rowv;
}

template<class T>
SGVector<T> SGMatrix<T>::get_diagonal_vector() const
{
	index_t diag_vlen=CMath::min(num_cols, num_rows);
	SGVector<T> diag(diag_vlen);

	for (int64_t i=0; i<diag_vlen; i++)
	{
		diag[i]=matrix[i*num_rows+i];
	}

	return diag;
}

template class SGMatrix<bool>;
template class SGMatrix<char>;
template class SGMatrix<int8_t>;
template class SGMatrix<uint8_t>;
template class SGMatrix<int16_t>;
template class SGMatrix<uint16_t>;
template class SGMatrix<int32_t>;
template class SGMatrix<uint32_t>;
template class SGMatrix<int64_t>;
template class SGMatrix<uint64_t>;
template class SGMatrix<float32_t>;
template class SGMatrix<float64_t>;
template class SGMatrix<floatmax_t>;
template class SGMatrix<complex128_t>;
}
