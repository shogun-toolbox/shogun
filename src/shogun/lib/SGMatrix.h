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
#ifndef __SGMATRIX_H__
#define __SGMATRIX_H__

#include <shogun/lib/config.h>
#include <shogun/lib/common.h>
#include <shogun/lib/SGReferencedData.h>

namespace Eigen
{
	template <class, int, int, int, int, int> class Matrix;
	template<int, int> class Stride;
	template <class, int, class> class Map;
}

namespace shogun
{
	template<class T> class SGVector;
	class CFile;

/** @brief shogun matrix */
template<class T> class SGMatrix : public SGReferencedData
{
	typedef Eigen::Matrix<T,-1,-1,0,-1,-1> EigenMatrixXt;
	typedef Eigen::Map<EigenMatrixXt,0,Eigen::Stride<0,0> > EigenMatrixXtMap;

	public:
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
		SGMatrix(T* m, index_t nrows, index_t ncols, index_t offset)
			: SGReferencedData(false), matrix(m+offset),
			num_rows(nrows), num_cols(ncols) { }

		/** Constructor to create new matrix in memory */
		SGMatrix(index_t nrows, index_t ncols, bool ref_counting=true);

#ifndef SWIG // SWIG should skip this part
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

#ifdef HAVE_EIGEN3
		/** Wraps a matrix around the data of an Eigen3 matrix */
		SGMatrix(EigenMatrixXt& mat);

		/** Wraps an Eigen3 matrix around the data of this matrix */
		operator EigenMatrixXtMap() const;
#endif // HAVE_EIGEN3
#endif // SWIG

		/** Copy constructor */
		SGMatrix(const SGMatrix &orig);

		/** Empty destructor */
		virtual ~SGMatrix();

#ifndef SWIG // SWIG should skip this part
		/** Get a column vector
		 * @param col column index
		 * @return the column vector for index col
		 */
		T* get_column_vector(index_t col) const
		{
			const int64_t c = col;
			return &matrix[c*num_rows];
		}

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
		    const int64_t c = i_col;
		    return matrix[c*num_rows + i_row];
		}

		/** Operator overload for matrix read only access
		 * @param index to access
		 */
		inline const T& operator[](index_t index) const
		{
			return matrix[index];
		}

		/** Operator overload for matrix r/w access
		 * @param i_row
		 * @param i_col
		 */
		inline T& operator()(index_t i_row, index_t i_col)
		{
		    const int64_t c = i_col;
		    return matrix[c*num_rows + i_row];
		}

		/** Operator overload for matrix r/w access
		 * @param index to access
		 */
		inline T& operator[](index_t index)
		{
			return matrix[index];
		}

		/**
		 * Get the matrix (no copying is done here)
		 *
		 * @return the refcount increased matrix
		 */
		inline SGMatrix<T> get()
		{
			return *this;
		}

		/** Check for pointer identity */
		bool operator==(SGMatrix<T>& other);

		/** Operator overload for element-wise matrix comparison.
		 * Note that only numerical data is compared
		 *
		 * @param other matrix to compare with
		 * @return true iff all elements are equal
		 */
		bool equals(SGMatrix<T>& other);

		/** Set matrix to a constant */
		void set_const(T const_elem);

		/** fill matrix with zeros */
		void zero();

		/**
		 * Checks whether the matrix is symmetric or not. The equality check
		 * is performed using '==' operators for discrete types (int, char,
		 * bool) and using CMath::fequals method for floating types (float,
		 * double, long double, std::complex<double>) with default espilon
		 * values from std::numeric_limits
		 *
		 * @return whether the matrix is symmetric
		 */
		bool is_symmetric();

		/** @return the maximum single element of the matrix */
		T max_single();

		/** Clone matrix */
		SGMatrix<T> clone();

		/** Clone matrix */
		static T* clone_matrix(const T* matrix, int32_t nrows, int32_t ncols);

		/** Transpose matrix */
		static void transpose_matrix(
			T*& matrix, int32_t& num_feat, int32_t& num_vec);

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

		/** Return the pseudo inverse for matrix
		 * when matrix has shape (rows, cols) the pseudo inverse has (cols, rows)
		 */
		static float64_t* pinv(
			float64_t* matrix, int32_t rows, int32_t cols,
			float64_t* target=NULL);

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
		void load(CFile* loader);

		/** Save matrix to file
		 *
		 * @param saver File object via which to save data
		 */
		void save(CFile* saver);
#endif // #ifndef SWIG // SWIG should skip this part

	protected:
		/** overridden to copy data */
		virtual void copy_data(const SGReferencedData &orig);

		/** overridden to initialize empty data */
		virtual void init_data();

		/** overridden to free data */
		virtual void free_data();

	public:
		/** matrix  */
		T* matrix;
		/** number of rows of matrix  */
		index_t num_rows;
		/** number of columns of matrix  */
		index_t num_cols;
};
}
#endif // __SGMATRIX_H__
