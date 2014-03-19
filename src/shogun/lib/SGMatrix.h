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

namespace shogun
{
	template<class T> class SGVector;
	class CFile;

/** @brief shogun matrix */
template<class T> class SGMatrix : public SGReferencedData
{
	public:
		/** default constructor */
		SGMatrix();

		/** constructor for setting reference counting while not creating
		 * the matrix in memory (use this for static SGMatrix instances) */
		SGMatrix(bool ref_counting);

		/** constructor for setting params */
		SGMatrix(T* m, index_t nrows, index_t ncols, bool ref_counting=true);

		/** constructor to create new matrix in memory */
		SGMatrix(index_t nrows, index_t ncols, bool ref_counting=true);

		/** copy constructor */
		SGMatrix(const SGMatrix &orig);

		/** empty destructor */
		virtual ~SGMatrix();

		/** get a column vector
		 * @param col column index
		 */
		T* get_column_vector(index_t col) const
		{
			const int64_t c = col;
			return &matrix[c*num_rows];
		}

		/** get a row vector
		 *
		 * @param row row index
		 * @return row vector
		 */
		SGVector<T> get_row_vector(index_t row) const;

		/** get a main diagonal vector. Matrix is not required to be square.
		 *
		 * @return main diagonal vector
		 */
		SGVector<T> get_diagonal_vector() const;

		/** operator overload for matrix read only access
		 * @param i_row
		 * @param i_col
		 */
		inline const T& operator()(index_t i_row, index_t i_col) const
		{
		    const int64_t c = i_col;
		    return matrix[c*num_rows + i_row];
		}

		/** operator overload for matrix read only access
		 * @param index to access
		 */
		inline const T& operator[](index_t index) const
		{
			return matrix[index];
		}

		/** operator overload for matrix r/w access
		 * @param i_row
		 * @param i_col
		 */
		inline T& operator()(index_t i_row, index_t i_col)
		{
		    const int64_t c = i_col;
		    return matrix[c*num_rows + i_row];
		}

		/** operator overload for matrix r/w access
		 * @param index to access
		 */
		inline T& operator[](index_t index)
		{
			return matrix[index];
		}

		/**
		 * get the matrix (no copying is done here)
		 *
		 * @return the refcount increased matrix
		 */
		inline SGMatrix<T> get()
		{
			return *this;
		}

		/** check for pointer identity */
		bool operator==(SGMatrix<T>& other);

		/** operator overload for element-wise matrix comparison.
		 * Note that only numerical data is compared
		 *
		 * @param other matrix to compare with
		 * @return true iff all elements are equal
		 */
		bool equals(SGMatrix<T>& other);

		/** set matrix to a constant */
		void set_const(T const_elem);

		/** fill matrix with zeros */
		void zero();

		/** returns the maximum single element of the matrix */
		T max_single();

		/** clone matrix */
		SGMatrix<T> clone();

		/** clone matrix */
		static T* clone_matrix(const T* matrix, int32_t nrows, int32_t ncols);

		/** transpose matrix */
		static void transpose_matrix(
			T*& matrix, int32_t& num_feat, int32_t& num_vec);

		/** create diagonal matrix */
		static void create_diagonal_matrix(T* matrix, T* v,int32_t size);

		/** returns the identity matrix, scaled by a factor
		 *
		 * @param size size of square identity matrix
		 * @param scale (optional) scaling factor
		 */
		static SGMatrix<T> create_identity_matrix(index_t size, T scale);

		/** returns the centering matrix, given by H=I-1/n*O, where
		 * I is the identity matrix, O is a square matrix of ones of size n
		 * Multiplied from the left hand side, subtracts from each column
		 * its mean.
		 * Multiplied from the right hand side, subtracts from each row
		 * its mean (so from each dimension of a SHOGUN feature)
		 *
		 * Note that H*H=H=H^T
		 *
		 * @param size size of centering matrix
		 */
		static SGMatrix<float64_t> create_centering_matrix(index_t size);

#ifdef HAVE_LAPACK
		/** compute eigenvalues and eigenvectors of symmetric matrix using
		 * LAPACK
		 *
		 * @param matrix symmetric matrix to compute eigenproblem. Is
		 * overwritten and contains orthonormal eigenvectors afterwards
		 * @return eigenvalues vector with eigenvalues equal to number of rows
		 * in matrix
		 * */
		static SGVector<float64_t> compute_eigenvectors(
				SGMatrix<float64_t> matrix);

		/** compute eigenvalues and eigenvectors of symmetric matrix
		 *
		 * @param matrix  overwritten and contains n orthonormal eigenvectors
		 * @param n
		 * @param m
		 * @return eigenvalues (array of length n, to be deleted[])
		 * */
		static double* compute_eigenvectors(double* matrix, int n, int m);

		/** compute few eigenpairs of a symmetric matrix using LAPACK DSYEVR method
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
		/** inverses square matrix in-place */
		static void inverse(SGMatrix<float64_t> matrix);

		/** return the pseudo inverse for matrix
		 * when matrix has shape (rows, cols) the pseudo inverse has (cols, rows)
		 */
		static float64_t* pinv(
			float64_t* matrix, int32_t rows, int32_t cols,
			float64_t* target=NULL);

#endif

		/** compute trace */
		static float64_t trace(
			float64_t* mat, int32_t cols, int32_t rows);

		/** sums up all rows of a matrix and returns the resulting rowvector */
		static T* get_row_sum(T* matrix, int32_t m, int32_t n);

		/** sums up all columns of a matrix and returns the resulting columnvector */
		static T* get_column_sum(T* matrix, int32_t m, int32_t n);

		/** Centers the matrix, i.e. removes column/row mean from columns/rows */
		void center();

		/** Centers  matrix (e.g. kernel matrix in feature space INPLACE */
		static void center_matrix(T* matrix, int32_t m, int32_t n);

		/** remove column mean */
		void remove_column_mean();

		/** display matrix */
		void display_matrix(const char* name="matrix") const;

		/** display matrix (useful for debugging) */
		static void display_matrix(
			const T* matrix, int32_t rows, int32_t cols,
			const char* name="matrix", const char* prefix="");

		/** display matrix */
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

		/** load matrix from file
		 *
		 * @param loader File object via which to load data
		 */
		void load(CFile* loader);

		/** save matrix to file
		 *
		 * @param saver File object via which to save data
		 */
		void save(CFile* saver);

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
