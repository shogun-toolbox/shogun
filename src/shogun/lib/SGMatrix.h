/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Soeren Sonnenburg, Heiko Strathmann, Soumyajit De, Sergey Lisitsyn,
 *          Pan Deng, Khaled Nasr, Michele Mazzoni, Viktor Gal,
 *          Fernando Iglesias, Thoralf Klein, Chiyuan Zhang, Koen van de Sande,
 *          Roman Votyakov
 */
#ifndef __SGMATRIX_H__
#define __SGMATRIX_H__

#include <shogun/base/macros.h>
#include <shogun/io/SGIO.h>
#include <shogun/lib/config.h>
#include <shogun/lib/common.h>
#include <shogun/util/iterators.h>
#include <shogun/lib/SGReferencedData.h>

#include <atomic>
#include <initializer_list>
#include <memory>

namespace Eigen
{
	template <class, int, int, int, int, int> class Matrix;
	template<int, int> class Stride;
	template <class, int, class> class Map;
}

namespace shogun
{
	template<class T> class SGVector;
	template<typename T> struct GPUMemoryBase;
	class File;

/** @brief shogun matrix */
template<class T> class SGMatrix : public SGReferencedData
{
	friend class LinalgBackendEigen;

	public:
		typedef RandomIterator<T> iterator;

	public:
		typedef Eigen::Matrix<T,-1,-1,0,-1,-1> EigenMatrixXt;
		typedef Eigen::Map<EigenMatrixXt,0,Eigen::Stride<0,0> > EigenMatrixXtMap;

		/** The scalar type of the matrix */
		typedef T Scalar;

		/** Default constructor */
		SGMatrix();

		/** Constructor for setting reference counting while not creating
		 * the matrix in memory (use this for static SGMatrix instances) */
		SGMatrix(bool ref_counting);

		/** Constructor for setting params */
		SGMatrix(T* m, index_t nrows, index_t ncols, bool ref_counting=true);

		/** Wraps a matrix around an existing memory segment with an offset */
		SGMatrix(T* m, index_t nrows, index_t ncols, index_t offset);

		/** Constructor to create new matrix in memory */
		SGMatrix(index_t nrows, index_t ncols, bool ref_counting=true);

		/**
		 * Construct SGMatrix from GPU memory.
		 *
		 * @param vector GPUMemoryBase pointer
		 * @param nrows row number of the matrix
		 * @param ncols column number of the matrix
		 * @see GPUMemoryBase
		 */
		 SGMatrix(GPUMemoryBase<T>* matrix, index_t nrows, index_t ncols);

		/** Check whether data is stored on GPU
		 *
		 * @return true if matrix is on GPU
		 */
#ifndef SWIG
		SG_FORCED_INLINE
#endif
		bool on_gpu() const
		{
			return gpu_ptr != NULL;
		}

#ifndef SWIG // SWIG should skip this part
		/** The container type for a given template argument */
		template <typename ST> using container_type = SGMatrix<ST>;

		/**
		 * Constructor for creating a SGMatrix from a SGVector with refcounting.
		 * We do not copy the data here, just the pointer to data and the ref-
		 * count object of the SGVector (i.e. vec and this share same data and
		 * ref-count object).
		 *
		 * This constructor assumes that the vector is the column 0 in the matrix.
		 *
		 * @param vec The SGVector
		 */
		SGMatrix(SGVector<T> vec);

		/**
		 * Constructor for creating a SGMatrix from a SGVector with refcounting.
		 * We do not copy the data here, just the pointer to data and the ref-
		 * count object of the SGVector (i.e. vec and this share same data and
		 * ref-count object).
		 *
		 * The number of elements in the matrix *MUST* be same as the number
		 * of elements in the vector
		 *
		 * @param vec The SGVector
		 * @param nrows number of rows in the matrix
		 * @param ncols number of columns in the matrix
		 */
		SGMatrix(SGVector<T> vec, index_t nrows, index_t ncols);

		/**
		 * Constructor for creating SGMatrix with an initializer list.
		 * As an example:
		 *  SGMatrix<T> m {{1.0, 1.0}, {2.0, 2.0}};
		 *
		 * Each element in the initializer list is a column vector as SGMatrix
		 * is column oriented!
		 *
		 * @param list initializer list of initializer lists
		 */
		SGMatrix(const std::initializer_list<std::initializer_list<T>>& list);

		/** Wraps a matrix around the data of an Eigen3 matrix */
		SGMatrix(EigenMatrixXt& mat);

		/** Wraps an Eigen3 matrix around the data of this matrix */
		operator EigenMatrixXtMap() const;

		/** Copy assign operator */
		SGMatrix<T>& operator=(const SGMatrix<T>&);
#endif // SWIG

		/** Copy constructor */
		SGMatrix(const SGMatrix &orig);

		/** Empty destructor */
		virtual ~SGMatrix();

#ifndef SWIG // SWIG should skip this parts
		/** Get a column vector
		 * @param col column index
		 * @return the column vector for index col
		 */
		T* get_column_vector(index_t col) const
		{
			assert_on_cpu();
			const int64_t c = col;
			return &matrix[c*num_rows];
		}

		/** Given a range of columns (start, end), return a view
		 * of the matrix from column start to end excluded.
		 * \warning The returned SGMatrix is non-owning!
		 * @param col_start column index (inclusive)
		 * @param col_end column index (excluded)
		 * @return the submatrix
		 */
		SGMatrix<T> slice(index_t col_start, index_t col_end) const;

		/** Map a column to a SGVector
		 * \warning The returned SGVector is non-owning!
		 * @param col column index
		 * @return the column vector for index col
		 */
		SGVector<T> get_column(index_t col) const;

		/** Copy the content of a vector into a column
		 * @param col column index
		 * @param vec vector
		 */
		void set_column(index_t col, const SGVector<T> vec);

		/** Get a row vector
		 *
		 * @param row row index
		 * @return row vector
		 */
		SGVector<T> get_row_vector(index_t row) const;

		/** Get a main diagonal vector. Matrix is not required to be square.
		 *
		 * @return main diagonal vector
		 */
		SGVector<T> get_diagonal_vector() const;

		/** Operator overload for matrix read only access
		 * @param i_row
		 * @param i_col
		 */
		inline const T& operator()(index_t i_row, index_t i_col) const
		{
			assert_on_cpu();
		    const int64_t c = i_col;
		    return matrix[c*num_rows + i_row];
		}

		/** Operator overload for matrix read only access
		 * @param index to access
		 */
		inline const T& operator[](index_t index) const
		{
			assert_on_cpu();
			return matrix[index];
		}

		/** Operator overload for matrix r/w access
		 * @param i_row
		 * @param i_col
		 */
		inline T& operator()(index_t i_row, index_t i_col)
		{
			assert_on_cpu();
		    const int64_t c = i_col;
		    return matrix[c*num_rows + i_row];
		}

		/** Operator overload for matrix r/w access
		 * @param index to access
		 */
		inline T& operator[](index_t index)
		{
			assert_on_cpu();
			return matrix[index];
		}

		/** Returns an iterator to the first element of the container. */
		iterator begin() noexcept { return iterator(matrix); }

		/** Returns an iterator to the element following the last element of the container. */
		iterator end() noexcept { return iterator(matrix + (num_rows * num_cols)); }

#endif // SWIG should skip this part

		/** Get element at index
		 *
		 * @param row row index
		 * @param col column index
		 * @return element at index
		 */
		const T& get_element(index_t row, index_t col)
		{
			return (*this)(row, col);
		}

		/** Set element at index
		 *
		 * @param el element to set
		 * @param row row index
		 * @param col column index
		 */
		void set_element(const T& el, index_t row, index_t col)
		{
			(*this)(row, col)=el;
		}

#ifndef SWIG // SWIG should skip this part

		/**
		 * Get the matrix (no copying is done here)
		 *
		 * @return the refcount increased matrix
		 */
		inline SGMatrix<T> get()
		{
			return *this;
		}

		/** The data */
		inline T* data() const
		{
			return matrix;
		}

		/** The size */
		inline int64_t size() const
		{
			const int64_t c=num_cols;
			return num_rows*c;
		}

		/** Check for pointer identity */
		SG_FORCED_INLINE bool operator==(const SGMatrix<T>& other) const
		{
			if (num_rows!=other.num_rows || num_cols!=other.num_cols)
				return false;

			if (on_gpu())
			{
				if (!other.on_gpu())
					return false;
				if (gpu_ptr!=other.gpu_ptr)
					return false;
			}

			if (matrix != other.matrix)
				return false;

			return true;
		}

		/** Operator overload for element-wise matrix comparison.
		 * Note that only numerical data is compared. Works for floating
		 * point numbers (along with complex128_t) as well.
		 *
		 * @param other matrix to compare with
		 * @return true iff all elements are equal
		 */
		bool equals(const SGMatrix<T>& other) const;

		/** Set matrix to a constant */
		void set_const(T const_elem);

		/** fill matrix with zeros */
		void zero();

		/**
		 * Checks whether the matrix is symmetric or not. The equality check
		 * is performed using '==' operators for discrete types (int, char,
		 * bool) and using Math::fequals method for floating types (float,
		 * double, long double, std::complex<double>) with default espilon
		 * values from std::numeric_limits
		 *
		 * @return whether the matrix is symmetric
		 */
		bool is_symmetric() const;

		/** @return the maximum single element of the matrix */
		T max_single() const;

		/** Clone matrix */
		SGMatrix<T> clone() const;

		/** Clone matrix */
		static T* clone_matrix(const T* matrix, int32_t nrows, int32_t ncols);

		/** Create diagonal matrix */
		static void create_diagonal_matrix(T* matrix, T* v,int32_t size);

		/** Returns the identity matrix, scaled by a factor
		 *
		 * @param size size of square identity matrix
		 * @param scale (optional) scaling factor
		 */
		static SGMatrix<T> create_identity_matrix(index_t size, T scale);

#ifdef HAVE_LAPACK
		/** Compute eigenvalues and eigenvectors of symmetric matrix using
		 * LAPACK
		 *
		 * @param matrix symmetric matrix to compute eigenproblem. Is
		 * overwritten and contains orthonormal eigenvectors afterwards
		 * @return eigenvalues vector with eigenvalues equal to number of rows
		 * in matrix
		 * */
		static SGVector<float64_t> compute_eigenvectors(
				SGMatrix<float64_t> matrix);

		/** Compute eigenvalues and eigenvectors of symmetric matrix
		 *
		 * @param matrix  overwritten and contains n orthonormal eigenvectors
		 * @param n
		 * @param m
		 * @return eigenvalues (array of length n, to be deleted[])
		 * */
		static double* compute_eigenvectors(double* matrix, int n, int m);

		/** Compute few eigenpairs of a symmetric matrix using LAPACK DSYEVR method
		 * (Relatively Robust Representations).
		 * Has at least O(n^3/3) complexity
		 * @param matrix_ symmetric matrix
		 * @param eigenvalues contains iu-il+1 eigenvalues in ascending order (to be free'd)
		 * @param eigenvectors contains iu-il+1 orthonormal eigenvectors of given matrix column-wise (to be free'd)
		 * @param n dimension of matrix
		 * @param il low index of requested eigenpairs (1<=il<=n)
		 * @param iu high index of requested eigenpairs (1<=il<=iu<=n)
		 */
		void compute_few_eigenvectors(double* matrix_, double*& eigenvalues, double*& eigenvectors,
                                      int n, int il, int iu);
#endif
		/** Computes scale* A*B, where A and B may be transposed.
		 *  Asserts for matching inner dimensions.
		 * @param A matrix A
		 * @param transpose_A optional whether A should be transposed before
		 * @param B matrix B
		 * @param transpose_B optional whether B should be transposed before
		 * @param scale optional scaling factor for result
		 */
		static SGMatrix<float64_t> matrix_multiply(
				SGMatrix<float64_t> A, SGMatrix<float64_t> B,
				bool transpose_A=false, bool transpose_B=false,
				float64_t scale=1.0);
#ifdef HAVE_LAPACK
		/** Inverses square matrix in-place */
		static void inverse(SGMatrix<float64_t> matrix);

#endif

		/** Compute trace */
		static float64_t trace(
			float64_t* mat, int32_t cols, int32_t rows);

		/** Sums up all rows of a matrix and returns the resulting rowvector */
		static T* get_row_sum(T* matrix, int32_t m, int32_t n);

		/** Sums up all columns of a matrix and returns the resulting columnvector */
		static T* get_column_sum(T* matrix, int32_t m, int32_t n);

		/** Centers the matrix, i.e. removes column/row mean from columns/rows */
		void center();

		/** Centers  matrix (e.g. kernel matrix in feature space INPLACE */
		static void center_matrix(T* matrix, int32_t m, int32_t n);

		/** Remove column mean */
		void remove_column_mean();

		/** String representation of the matrix */
		std::string to_string() const;

		/** String representation of the matrix */
		static std::string to_string(
			const T* matrix, index_t rows, index_t cols);

		/** Display matrix */
		void display_matrix(const char* name="matrix") const;

		/** Display matrix (useful for debugging) */
		static void display_matrix(
			const T* matrix, int32_t rows, int32_t cols,
			const char* name="matrix", const char* prefix="");

		/** Display matrix */
		static void display_matrix(
			const SGMatrix<T> matrix, const char* name="matrix",
			const char* prefix="");

		/** Simple helper method that returns a matrix with allocated memory
		 * for a given size. A pre_allocated one can optionally be specified
		 * in order to use that.
		 * Basically just for having dimension check encapsulated.
		 *
		 * @param num_rows rows of returned matrix
		 * @param num_cols columns of returned matrix
		 * @param pre_allocated optional matrix that is returned instead of new
		 * matrix. Make sure dimensions match
		 * @return matrix with allocated memory of specified size
		 */
		static SGMatrix<T> get_allocated_matrix(index_t num_rows,
				index_t num_cols, SGMatrix<T> pre_allocated=SGMatrix<T>());

		/** Load matrix from file
		 *
		 * @param loader File object via which to load data
		 */
		void load(const std::shared_ptr<File>& loader);

		/** Save matrix to file
		 *
		 * @param saver File object via which to save data
		 */
		void save(const std::shared_ptr<File>& saver);
#endif // #ifndef SWIG // SWIG should skip this part

	protected:
		/** overridden to copy data */
		virtual void copy_data(const SGReferencedData &orig);

		/** overridden to initialize empty data */
		virtual void init_data();

		/** overridden to free data */
		virtual void free_data();

  private:
		/** Atomic variable of vector on_gpu status */
		std::atomic<bool> m_on_gpu;

		/** Assert whether the data is on GPU
		 * and raise error if the data is on GPU
		 */
#ifndef SWIG
		SG_FORCED_INLINE
#endif
		void assert_on_cpu() const
		{
			if (on_gpu())
				error("Direct memory access not possible when data is in GPU memory.");
		}

	public:
		/** matrix  */
		T* matrix;
		/** number of rows of matrix  */
		index_t num_rows;
		/** number of columns of matrix  */
		index_t num_cols;
		/** GPU Matrix structure. Stores pointer to the data on GPU. */
		std::shared_ptr<GPUMemoryBase<T>> gpu_ptr;
};
}
#endif // __SGMATRIX_H__
