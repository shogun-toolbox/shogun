/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Heiko Strathmann, Soeren Sonnenburg, Soumyajit De, Thoralf Klein,
 *          Pan Deng, Fernando Iglesias, Sergey Lisitsyn, Viktor Gal,
 *          Michele Mazzoni, Yingrui Chang, Weijie Lin, Khaled Nasr,
 *          Koen van de Sande, Roman Votyakov
 */

#include <shogun/lib/config.h>
#include <shogun/io/SGIO.h>
#include <shogun/io/File.h>
#include <shogun/lib/SGMatrix.h>
#include <shogun/lib/SGVector.h>
#include <shogun/mathematics/eigen3.h>
#include <shogun/mathematics/Math.h>
#include <shogun/mathematics/lapack.h>
#include <limits>
#include <algorithm>

namespace shogun
{

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
	num_rows(nrows), num_cols(ncols), gpu_ptr(nullptr)
{
	m_on_gpu.store(false, std::memory_order_release);
}

template <class T>
SGMatrix<T>::SGMatrix(T* m, index_t nrows, index_t ncols, index_t offset)
	: SGReferencedData(false), matrix(m+offset),
	num_rows(nrows), num_cols(ncols)
{
	m_on_gpu.store(false, std::memory_order_release);
}

template <class T>
SGMatrix<T>::SGMatrix(index_t nrows, index_t ncols, bool ref_counting)
	: SGReferencedData(ref_counting), num_rows(nrows), num_cols(ncols), gpu_ptr(nullptr)
{
	matrix=SG_ALIGNED_MALLOC(
		T, ((int64_t) nrows)*ncols, alignment::container_alignment);
	std::fill_n(matrix, ((int64_t) nrows)*ncols, 0);
	m_on_gpu.store(false, std::memory_order_release);
}

template <class T>
SGMatrix<T>::SGMatrix(SGVector<T> vec) : SGReferencedData(vec)
{
	require((vec.vector || vec.gpu_ptr), "Vector not initialized!");
	matrix=vec.vector;
	num_rows=vec.vlen;
	num_cols=1;
	gpu_ptr = vec.gpu_ptr;
	m_on_gpu.store(vec.on_gpu(), std::memory_order_release);
}

template <class T>
SGMatrix<T>::SGMatrix(SGVector<T> vec, index_t nrows, index_t ncols)
: SGReferencedData(vec)
{
	require((vec.vector || vec.gpu_ptr), "Vector not initialized!");
	require(nrows>0, "Number of rows ({}) has to be a positive integer!", nrows);
	require(ncols>0, "Number of cols ({}) has to be a positive integer!", ncols);
	require(vec.vlen==nrows*ncols, "Number of elements in the matrix ({}) must "
			"be the same as the number of elements in the vector ({})!",
			nrows*ncols, vec.vlen);

	matrix=vec.vector;
	num_rows=nrows;
	num_cols=ncols;
	gpu_ptr = vec.gpu_ptr;
	m_on_gpu.store(vec.on_gpu(), std::memory_order_release);
}

template<class T>
SGMatrix<T>::SGMatrix(GPUMemoryBase<T>* mat, index_t nrows, index_t ncols)
	: SGReferencedData(true), matrix(NULL), num_rows(nrows), num_cols(ncols),
	gpu_ptr(std::shared_ptr<GPUMemoryBase<T>>(mat))
{
	m_on_gpu.store(true, std::memory_order_release);
}

template <class T>
SGMatrix<T>::SGMatrix(const SGMatrix &orig) : SGReferencedData(orig)
{
	copy_data(orig);
}

template <class T>
SGMatrix<T>::SGMatrix(EigenMatrixXt& mat)
: SGReferencedData(false), matrix(mat.data()),
	num_rows(mat.rows()), num_cols(mat.cols()), gpu_ptr(nullptr)
{
	m_on_gpu.store(false, std::memory_order_release);
}

template <class T>
SGMatrix<T>::SGMatrix(const std::initializer_list<std::initializer_list<T>>& list):
	SGReferencedData(),
	num_rows((*list.begin()).size()),
	num_cols(list.size()),
	gpu_ptr(nullptr)
{
	matrix = SG_CALLOC(T, ((int64_t) num_rows)*num_cols);
	m_on_gpu.store(false, std::memory_order_release);
	int64_t curr_pos = 0;
	for (const auto& r: list)
		for (const auto& c: r)
			matrix[curr_pos++] = c;
}

template <class T>
SGMatrix<T>::operator EigenMatrixXtMap() const
{
	assert_on_cpu();
	return EigenMatrixXtMap(matrix, num_rows, num_cols);
}

template<class T>
SGMatrix<T>& SGMatrix<T>::operator=(const SGMatrix<T>& other)
{
	if(&other == this)
	return *this;

	unref();
	copy_data(other);
	copy_refcount(other);
	ref();
	return *this;
}

template <class T>
SGMatrix<T>::~SGMatrix()
{
	unref();
}

template <class T>
bool SGMatrix<T>::equals(const SGMatrix<T>& other) const
{
	// avoid comparing elements when both are same.
	// the case where both matrices are uninitialized is handled here as well.
	if (*this==other)
		return true;

	// both empty
	if (!(num_rows || num_cols || other.num_rows || other.num_cols))
		return true;

	// only one empty
	if (!matrix || !other.matrix)
		return false;

	// different size
	if (num_rows!=other.num_rows || num_cols!=other.num_cols)
		return false;

	// content
	return std::equal(matrix, matrix+size(), other.matrix);
}

#ifndef REAL_EQUALS
#define REAL_EQUALS(real_t)                                                    \
	template <>                                                                \
	bool SGMatrix<real_t>::equals(const SGMatrix<real_t>& other) const         \
	{                                                                          \
		if (*this == other)                                                    \
			return true;                                                       \
                                                                               \
		if (!(num_rows || num_cols || other.num_rows || other.num_cols))       \
			return true;                                                       \
                                                                               \
		if (!matrix || !other.matrix)                                          \
			return false;                                                      \
                                                                               \
		if (num_rows != other.num_rows || num_cols != other.num_cols)          \
			return false;                                                      \
                                                                               \
		return std::equal(                                                     \
		    matrix, matrix + size(), other.matrix,                             \
		    [](const real_t& a, const real_t& b) {                             \
			    return Math::fequals<real_t>(                                 \
			        a, b, std::numeric_limits<real_t>::epsilon());             \
			});                                                                \
	}

REAL_EQUALS(float32_t)
REAL_EQUALS(float64_t)
REAL_EQUALS(floatmax_t)
#undef REAL_EQUALS
#endif // REAL_EQUALS

template <>
bool SGMatrix<complex128_t>::equals(const SGMatrix<complex128_t>& other) const
{
	if (*this==other)
		return true;

	if (matrix==nullptr || other.matrix==nullptr)
		return false;

	if (num_rows!=other.num_rows || num_cols!=other.num_cols)
		return false;

	return std::equal(matrix, matrix+size(), other.matrix,
		[](const complex128_t& a, const complex128_t& b)
		{
			return Math::fequals<float64_t>(a.real(), b.real(), LDBL_EPSILON) &&
				Math::fequals<float64_t>(a.imag(), b.imag(), LDBL_EPSILON);
		});
}

template <class T>
void SGMatrix<T>::set_const(T const_elem)
{
	assert_on_cpu();

	require(matrix!=nullptr, "The underlying matrix is not allocated!");
	require(num_rows>0, "Number of rows ({}) has to be positive!", num_rows);
	require(num_cols>0, "Number of cols ({}) has to be positive!", num_cols);

	std::fill(matrix, matrix+size(), const_elem);
}

template <class T>
void SGMatrix<T>::zero()
{
	set_const(static_cast<T>(0));
}

template <class T>
bool SGMatrix<T>::is_symmetric() const
{
	assert_on_cpu();

	require(matrix!=nullptr, "The underlying matrix is not allocated!");
	require(num_rows>0, "Number of rows ({}) has to be positive!", num_rows);
	require(num_cols>0, "Number of cols ({}) has to be positive!", num_cols);

	if (num_rows!=num_cols)
		return false;

	for (index_t i=0; i<num_rows; ++i)
	{
		for (index_t j=i+1; j<num_cols; ++j)
		{
			if (matrix[j*num_rows+i]!=matrix[i*num_rows+j])
				return false;
		}
	}

	return true;
}

#ifndef REAL_IS_SYMMETRIC
#define REAL_IS_SYMMETRIC(real_t)	\
template <>	\
bool SGMatrix<real_t>::is_symmetric() const	\
{	\
	assert_on_cpu();	\
	\
	require(matrix!=nullptr, "The underlying matrix is not allocated!");	\
	require(num_rows>0, "Number of rows ({}) has to be positive!", num_rows);	\
	require(num_cols>0, "Number of cols ({}) has to be positive!", num_cols);	\
	\
	if (num_rows!=num_cols)	\
		return false;	\
	\
	for (index_t i=0; i<num_rows; ++i)	\
	{	\
		for (index_t j=i+1; j<num_cols; ++j)	\
		{	\
			if (!Math::fequals<real_t>(matrix[j*num_rows+i],	\
						matrix[i*num_rows+j], std::numeric_limits<real_t>::epsilon()))	\
				return false;	\
		}	\
	}	\
	\
	return true;	\
}

REAL_IS_SYMMETRIC(float32_t)
REAL_IS_SYMMETRIC(float64_t)
REAL_IS_SYMMETRIC(floatmax_t)
#undef REAL_IS_SYMMETRIC
#endif // REAL_IS_SYMMETRIC

template <>
bool SGMatrix<complex128_t>::is_symmetric() const
{
	assert_on_cpu();

	require(matrix!=nullptr, "The underlying matrix is not allocated!");
	require(num_rows>0, "Number of rows ({}) has to be positive!", num_rows);
	require(num_cols>0, "Number of cols ({}) has to be positive!", num_cols);

	if (num_rows!=num_cols)
		return false;

	for (index_t i=0; i<num_rows; ++i)
	{
		for (index_t j=i+1; j<num_cols; ++j)
		{
			if (!(Math::fequals<float64_t>(matrix[j*num_rows+i].real(),
						matrix[i*num_rows+j].real(), DBL_EPSILON) &&
					Math::fequals<float64_t>(matrix[j*num_rows+i].imag(),
						matrix[i*num_rows+j].imag(), DBL_EPSILON)))
				return false;
		}
	}

	return true;
}

template <class T>
T SGMatrix<T>::max_single() const
{
	assert_on_cpu();

	require(matrix!=nullptr, "The underlying matrix is not allocated!");
	require(num_rows>0, "Number of rows ({}) has to be positive!", num_rows);
	require(num_cols>0, "Number of cols ({}) has to be positive!", num_cols);

	return *std::max_element(matrix, matrix+size());
}

template <>
complex128_t SGMatrix<complex128_t>::max_single() const
{
	error("SGMatrix::max_single():: Not supported for complex128_t");
	return complex128_t(0.0);
}

template <class T>
SGMatrix<T> SGMatrix<T>::clone() const
{
	if (on_gpu())
	{
		return SGMatrix<T>(gpu_ptr->clone_vector(gpu_ptr.get(),
						   num_rows*num_cols), num_rows, num_cols);
	}
	else
	{
		return SGMatrix<T>(clone_matrix(matrix, num_rows, num_cols),
						   num_rows, num_cols);
	}
}

template <class T>
T* SGMatrix<T>::clone_matrix(const T* matrix, int32_t nrows, int32_t ncols)
{
	if (!matrix || !nrows || !ncols)
		return nullptr;

	require(nrows > 0, "Number of rows ({}) has to be positive!", nrows);
	require(ncols > 0, "Number of cols ({}) has to be positive!", ncols);

	auto size=int64_t(nrows)*ncols;
	T* result=SG_ALIGNED_MALLOC(T, size, alignment::container_alignment);
	sg_memcpy(result, matrix, size*sizeof(T));
	return result;
}

template <class T>
void SGMatrix<T>::create_diagonal_matrix(T* matrix, T* v,int32_t size)
{
	/* Need assert v.size() */
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
SGMatrix<T> SGMatrix<T>::slice(index_t col_start, index_t col_end) const
{
	assert_on_cpu();
	return SGMatrix<T>(
		get_column_vector(col_start), num_rows, col_end - col_start, false);
}

template <class T>
SGVector<T> SGMatrix<T>::get_column(index_t col) const
{
	assert_on_cpu();
	return SGVector<T>(get_column_vector(col), num_rows, false);
}

template <class T>
void SGMatrix<T>::set_column(index_t col, const SGVector<T> vec)
{
	assert_on_cpu();
	ASSERT(!vec.on_gpu())
	ASSERT(vec.vlen == num_rows)
	sg_memcpy(&matrix[num_rows * col], vec.vector, sizeof(T) * num_rows);
}

template <class T>
float64_t SGMatrix<T>::trace(float64_t* mat, int32_t cols, int32_t rows)
{
	float64_t trace=0;
	for (int64_t i=0; i<rows; i++)
		trace+=mat[i*cols+i];
	return trace;
}

/* Already in linalg */
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

/* Already in linalg */
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
	assert_on_cpu();
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
	assert_on_cpu();

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

template<class T>
std::string SGMatrix<T>::to_string() const
{
	assert_on_cpu();
	return to_string(matrix, num_rows, num_cols);
}

template<class T>
std::string SGMatrix<T>::to_string(const T* matrix, index_t rows, index_t cols)
{
	std::stringstream ss;
	ss << std::boolalpha << "[\n";
	if (rows != 0 && cols != 0)
	{
		for (int64_t i=0; i<rows; ++i)
		{
			ss << "[";
			for (int64_t j=0; j<cols; ++j)
				ss << "\t" << matrix[j*rows+i]
					<< (j == cols - 1 ? "" : ",");
			ss << "]" << (i == rows-1 ? "" : ",") << "\n";
		}
	}
	ss << "]";
	return ss.str();
}

template<class T> void SGMatrix<T>::display_matrix(const char* name) const
{
	assert_on_cpu();
	display_matrix(matrix, num_rows, num_cols, name);
}

template <class T>
void SGMatrix<T>::display_matrix(
	const SGMatrix<T> matrix, const char* name,
	const char* prefix)
{
	matrix.display_matrix();
}

template <class T>
void SGMatrix<T>::display_matrix(
	const T* matrix, int32_t rows, int32_t cols, const char* name,
	const char* prefix)
{
	ASSERT(rows>=0 && cols>=0)
	io::print("{}{}={}\n", prefix, name, to_string(matrix, rows, cols).c_str());
}

template <>
SGMatrix<char> SGMatrix<char>::create_identity_matrix(index_t size, char scale)
{
	not_implemented(SOURCE_LOCATION);
	return SGMatrix<char>();
}

template <>
SGMatrix<int8_t> SGMatrix<int8_t>::create_identity_matrix(index_t size, int8_t scale)
{
	SGMatrix<int8_t> identity_matrix(size, size);
	for (index_t i=0; i<size; ++i)
	{
		for (index_t j=0; j<size; ++j)
			identity_matrix(i,j)=i==j ? scale : 0.0;
	}

	return identity_matrix;
}

template <>
SGMatrix<uint8_t> SGMatrix<uint8_t>::create_identity_matrix(index_t size, uint8_t scale)
{
	SGMatrix<uint8_t> identity_matrix(size, size);
	for (index_t i=0; i<size; ++i)
	{
		for (index_t j=0; j<size; ++j)
			identity_matrix(i,j)=i==j ? scale : 0.0;
	}

	return identity_matrix;
}

template <>
SGMatrix<bool> SGMatrix<bool>::create_identity_matrix(index_t size, bool scale)
{
	SGMatrix<bool> identity_matrix(size, size);
	for (index_t i=0; i<size; ++i)
	{
		for (index_t j=0; j<size; ++j)
			identity_matrix(i,j)=i==j ? scale : (!scale);
	}

	return identity_matrix;
}

template <>
SGMatrix<int16_t> SGMatrix<int16_t>::create_identity_matrix(index_t size, int16_t scale)
{
	SGMatrix<int16_t> identity_matrix(size, size);
	for (index_t i=0; i<size; ++i)
	{
		for (index_t j=0; j<size; ++j)
			identity_matrix(i,j)=i==j ? scale : 0.0;
	}

	return identity_matrix;
}

template <>
SGMatrix<uint16_t> SGMatrix<uint16_t>::create_identity_matrix(index_t size, uint16_t scale)
{
	SGMatrix<uint16_t> identity_matrix(size, size);
	for (index_t i=0; i<size; ++i)
	{
		for (index_t j=0; j<size; ++j)
			identity_matrix(i,j)=i==j ? scale : 0.0;
	}

	return identity_matrix;
}

template <>
SGMatrix<int32_t> SGMatrix<int32_t>::create_identity_matrix(index_t size, int32_t scale)
{
	SGMatrix<int32_t> identity_matrix(size, size);
	for (index_t i=0; i<size; ++i)
	{
		for (index_t j=0; j<size; ++j)
			identity_matrix(i,j)=i==j ? scale : 0.0;
	}

	return identity_matrix;
}

template <>
SGMatrix<uint32_t> SGMatrix<uint32_t>::create_identity_matrix(index_t size, uint32_t scale)
{
	SGMatrix<uint32_t> identity_matrix(size, size);
	for (index_t i=0; i<size; ++i)
	{
		for (index_t j=0; j<size; ++j)
			identity_matrix(i,j)=i==j ? scale : 0.0;
	}

	return identity_matrix;
}

template <>
SGMatrix<int64_t> SGMatrix<int64_t>::create_identity_matrix(index_t size, int64_t scale)
{
	SGMatrix<int64_t> identity_matrix(size, size);
	for (index_t i=0; i<size; ++i)
	{
		for (index_t j=0; j<size; ++j)
			identity_matrix(i,j)=i==j ? scale : 0.0;
	}

	return identity_matrix;
}

template <>
SGMatrix<uint64_t> SGMatrix<uint64_t>::create_identity_matrix(index_t size, uint64_t scale)
{
	SGMatrix<uint64_t> identity_matrix(size, size);
	for (index_t i=0; i<size; ++i)
	{
		for (index_t j=0; j<size; ++j)
			identity_matrix(i,j)=i==j ? scale : 0.0;
	}

	return identity_matrix;
}

template <>
SGMatrix<float32_t> SGMatrix<float32_t>::create_identity_matrix(index_t size, float32_t scale)
{
	SGMatrix<float32_t> identity_matrix(size, size);
	for (index_t i=0; i<size; ++i)
	{
		for (index_t j=0; j<size; ++j)
			identity_matrix(i,j)=i==j ? scale : 0.0;
	}

	return identity_matrix;
}

template <>
SGMatrix<float64_t> SGMatrix<float64_t>::create_identity_matrix(index_t size, float64_t scale)
{
	SGMatrix<float64_t> identity_matrix(size, size);
	for (index_t i=0; i<size; ++i)
	{
		for (index_t j=0; j<size; ++j)
			identity_matrix(i,j)=i==j ? scale : 0.0;
	}

	return identity_matrix;
}

template <>
SGMatrix<floatmax_t> SGMatrix<floatmax_t>::create_identity_matrix(index_t size, floatmax_t scale)
{
	SGMatrix<floatmax_t> identity_matrix(size, size);
	for (index_t i=0; i<size; ++i)
	{
		for (index_t j=0; j<size; ++j)
			identity_matrix(i,j)=i==j ? scale : 0.0;
	}

	return identity_matrix;
}

template <>
SGMatrix<complex128_t> SGMatrix<complex128_t>::create_identity_matrix(index_t size, complex128_t scale)
{
	SGMatrix<complex128_t> identity_matrix(size, size);
	for (index_t i=0; i<size; ++i)
	{
		for (index_t j=0; j<size; ++j)
			identity_matrix(i,j)=i==j ? scale : complex128_t(0.0);
	}

	return identity_matrix;
}

#ifdef HAVE_LAPACK
/// inverses square matrix in-place
template <class T>
void SGMatrix<T>::inverse(SGMatrix<float64_t> matrix)
{
	require(!matrix.on_gpu(), "Operation is not possible when data is in GPU memory.");
	ASSERT(matrix.num_cols==matrix.num_rows);
	int32_t* ipiv = SG_MALLOC(int32_t, matrix.num_cols);
	clapack_dgetrf(CblasColMajor,matrix.num_cols,matrix.num_cols,matrix.matrix,matrix.num_cols,ipiv);
	clapack_dgetri(CblasColMajor,matrix.num_cols,matrix.matrix,matrix.num_cols,ipiv);
	SG_FREE(ipiv);
}

template <class T>
SGVector<float64_t> SGMatrix<T>::compute_eigenvectors(SGMatrix<float64_t> matrix)
{
	require(!matrix.on_gpu(), "Operation is not possible when data is in GPU memory.");
	if (matrix.num_rows!=matrix.num_cols)
	{
		error("SGMatrix::compute_eigenvectors(SGMatrix<float64_t>): matrix"
				" rows and columns are not equal!");
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
		error("DSYEV failed with code {}", info);

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

/* Already in linalg */
template <class T>
SGMatrix<float64_t> SGMatrix<T>::matrix_multiply(
		SGMatrix<float64_t> A, SGMatrix<float64_t> B,
		bool transpose_A, bool transpose_B, float64_t scale)
{
	require((!A.on_gpu()) && (!B.on_gpu()),
		"Operation is not possible when data is in GPU memory.");

	/* these variables store size of transposed matrices*/
	index_t cols_A=transpose_A ? A.num_rows : A.num_cols;
	index_t rows_A=transpose_A ? A.num_cols : A.num_rows;
	index_t rows_B=transpose_B ? B.num_cols : B.num_rows;
	index_t cols_B=transpose_B ? B.num_rows : B.num_cols;

	/* do a dimension check */
	if (cols_A!=rows_B)
	{
		error("SGMatrix::matrix_multiply(): Dimension mismatch: "
				"A({}x{})*B({}x%D)", rows_A, cols_A, rows_B, cols_B);
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
	if (pre_allocated.matrix || pre_allocated.gpu_ptr)
	{
		result=pre_allocated;

		/* check dimension consistency */
		if (pre_allocated.num_rows!=num_rows ||
				pre_allocated.num_cols!=num_cols)
		{
			error("SGMatrix<T>::get_allocated_matrix(). Provided target"
					"matrix dimensions ({}x{}) do not match passed data "
					"dimensions ({}x{})!", pre_allocated.num_rows,
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
	gpu_ptr=((SGMatrix*)(&orig))->gpu_ptr;
	matrix=((SGMatrix*)(&orig))->matrix;
	num_rows=((SGMatrix*)(&orig))->num_rows;
	num_cols=((SGMatrix*)(&orig))->num_cols;
	m_on_gpu.store(((SGMatrix*)(&orig))->m_on_gpu.load(
		std::memory_order_acquire), std::memory_order_release);
}

template<class T>
void SGMatrix<T>::init_data()
{
	matrix=NULL;
	num_rows=0;
	num_cols=0;
	gpu_ptr=nullptr;
	m_on_gpu.store(false, std::memory_order_release);
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
void SGMatrix<T>::load(const std::shared_ptr<File>& loader)
{
	ASSERT(loader)
	unref();

	SG_SET_LOCALE_C;
	SGMatrix<T> mat;
	loader->get_matrix(mat.matrix, mat.num_rows, mat.num_cols);
	mat.gpu_ptr = nullptr;
	copy_data(mat);
	copy_refcount(mat);
	ref();
	SG_RESET_LOCALE;
}

template<>
void SGMatrix<complex128_t>::load(const std::shared_ptr<File>& loader)
{
	error("SGMatrix::load():: Not supported for complex128_t");
}

template<class T>
void SGMatrix<T>::save(const std::shared_ptr<File>& writer)
{
	assert_on_cpu();
	ASSERT(writer)
	SG_SET_LOCALE_C;
	writer->set_matrix(matrix, num_rows, num_cols);
	SG_RESET_LOCALE;
}

template<>
void SGMatrix<complex128_t>::save(const std::shared_ptr<File>& saver)
{
	error("SGMatrix::save():: Not supported for complex128_t");
}

template<class T>
SGVector<T> SGMatrix<T>::get_row_vector(index_t row) const
{
	assert_on_cpu();
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
	assert_on_cpu();
	index_t diag_vlen=Math::min(num_cols, num_rows);
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
