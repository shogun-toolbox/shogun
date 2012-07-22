/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2012 Fernando José Iglesias García
 * Written (W) 2010,2012 Soeren Sonnenburg
 * Copyright (C) 2010 Berlin Institute of Technology
 * Copyright (C) 2012 Soeren Sonnenburg
 */
#ifndef __SGMATRIX_H__
#define __SGMATRIX_H__

#include <shogun/lib/config.h>
#include <shogun/lib/DataType.h>
#include <shogun/lib/SGReferencedData.h>

namespace shogun
{
	template<class T> class SGVector;
/** @brief shogun matrix */
template<class T> class SGMatrix : public SGReferencedData
{
	public:
		/** default constructor */
		SGMatrix() : SGReferencedData(false)
		{
			init_data();
		}

		/** constructor for setting params */
		SGMatrix(T* m, index_t nrows, index_t ncols, bool ref_counting=true)
			: SGReferencedData(ref_counting), matrix(m),
			num_rows(nrows), num_cols(ncols) { }

		/** constructor to create new matrix in memory */
		SGMatrix(index_t nrows, index_t ncols, bool ref_counting=true)
			: SGReferencedData(ref_counting), num_rows(nrows), num_cols(ncols)
		{
			matrix=SG_MALLOC(T, ((int64_t) nrows)*ncols);
		}

		/** copy constructor */
		SGMatrix(const SGMatrix &orig) : SGReferencedData(orig)
		{
			copy_data(orig);
		}

		/** empty destructor */
		virtual ~SGMatrix()
		{
			unref();
		}

		/** get a column vector
		 * @param col column index
		 */
		T* get_column_vector(index_t col) const
		{
			return &matrix[col*num_rows];
		}

		/** operator overload for matrix read only access
		 * @param i_row
		 * @param i_col
		 */
		inline const T& operator()(index_t i_row, index_t i_col) const
		{
		    return matrix[i_col*num_rows + i_row];
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
		    return matrix[i_col*num_rows + i_row];
		}

		/** operator overload for matrix r/w access
		 * @param index to access
		 */
		inline T& operator[](index_t index)
		{
			return matrix[index];
		}

		/** set matrix to a constant */
		void set_const(T const_elem)
		{
			for (index_t i=0; i<num_rows*num_cols; i++)
				matrix[i]=const_elem ;
		}

		/** fill matrix with zeros */
		void zero()
		{
			if (matrix && (num_rows*num_cols))
				set_const(0);
		}

		/** clone matrix */
		SGMatrix<T> clone()
		{
			return SGMatrix<T>(clone_matrix(matrix, num_rows, num_cols),
					num_rows, num_cols);
		}

		/** clone vector */
		static T* clone_matrix(const T* matrix, int32_t nrows, int32_t ncols)
		{
			T* result = SG_MALLOC(T, int64_t(nrows)*ncols);
			for (int64_t i=0; i<int64_t(nrows)*ncols; i++)
				result[i]=matrix[i];

			return result;
		}

		static void transpose_matrix(
			T*& matrix, int32_t& num_feat, int32_t& num_vec);

		static void create_diagonal_matrix(T* matrix, T* v,int32_t size)
		{
			for(int32_t i=0;i<size;i++)
			{
				for(int32_t j=0;j<size;j++)
				{
					if(i==j)
						matrix[j*size+i]=v[i];
					else
						matrix[j*size+i]=0;
				}
			}
		}

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

		/* Computes scale* A*B, where A and B may be transposed.
		 * Asserts for matching inner dimensions.
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

		/// inverses square matrix in-place
		static void inverse(SGMatrix<float64_t> matrix);

		/// return the pseudo inverse for matrix
		/// when matrix has shape (rows, cols) the pseudo inverse has (cols, rows)
		static float64_t* pinv(
			float64_t* matrix, int32_t rows, int32_t cols,
			float64_t* target=NULL);

#endif

		static inline float64_t trace(
			float64_t* mat, int32_t cols, int32_t rows)
		{
			float64_t trace=0;
			for (int32_t i=0; i<rows; i++)
				trace+=mat[i*cols+i];
			return trace;
		}

		/* Sums up all rows of a matrix and returns the resulting rowvector */
		static T* get_row_sum(T* matrix, int32_t m, int32_t n)
		{
			T* rowsums=SG_CALLOC(T, n);

			for (int32_t i=0; i<n; i++)
			{
				for (int32_t j=0; j<m; j++)
					rowsums[i]+=matrix[j+int64_t(i)*m];
			}
			return rowsums;
		}

		/* Sums up all columns of a matrix and returns the resulting columnvector */
		static T* get_column_sum(T* matrix, int32_t m, int32_t n)
		{
			T* colsums=SG_CALLOC(T, m);

			for (int32_t i=0; i<n; i++)
			{
				for (int32_t j=0; j<m; j++)
					colsums[j]+=matrix[j+int64_t(i)*m];
			}
			return colsums;
		}

		/** Centers the matrix, i.e. removes column/row mean from columns/rows */
		void center()
		{
			center_matrix(matrix, num_rows, num_cols);
		}

		/* Centers  matrix (e.g. kernel matrix in feature space INPLACE */
		static void center_matrix(T* matrix, int32_t m, int32_t n);

		void remove_column_mean();

		/** display matrix */
		void display_matrix(const char* name="matrix") const;

		/// display matrix (useful for debugging)
		static void display_matrix(
			const T* matrix, int32_t rows, int32_t cols,
			const char* name="matrix", const char* prefix="");

		static void display_matrix(
			const SGMatrix<T> matrix, const char* name="matrix",
			const char* prefix="");

	protected:
		/** needs to be overridden to copy data */
		virtual void copy_data(const SGReferencedData &orig)
		{
			matrix=((SGMatrix*)(&orig))->matrix;
			num_rows=((SGMatrix*)(&orig))->num_rows;
			num_cols=((SGMatrix*)(&orig))->num_cols;
		}

		/** needs to be overridden to initialize empty data */
		virtual void init_data()
		{
			matrix=NULL;
			num_rows=0;
			num_cols=0;
		}

		/** needs to be overridden to free data */
		virtual void free_data()
		{
			SG_FREE(matrix);
			matrix=NULL;
			num_rows=0;
			num_cols=0;
		}

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
