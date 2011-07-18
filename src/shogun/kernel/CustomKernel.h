/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 1999-2009 Soeren Sonnenburg
 * Copyright (C) 1999-2009 Fraunhofer Institute FIRST and Max-Planck-Society
 */

#ifndef _CUSTOMKERNEL_H___
#define _CUSTOMKERNEL_H___

#include <shogun/lib/Mathematics.h>
#include <shogun/lib/common.h>
#include <shogun/kernel/Kernel.h>
#include <shogun/features/Features.h>

namespace shogun
{
/** @brief The Custom Kernel allows for custom user provided kernel matrices.
 *
 * For squared training matrices it allows to store only the upper triangle of
 * the kernel to save memory: Full symmetric kernel matrices can be stored as
 * is or can be internally converted into (or directly given in) upper triangle
 * representation. Also note that values are stored as 32bit floats.
 *
 */
class CCustomKernel: public CKernel
{
	void init(void);

	public:
		/** default constructor */
		CCustomKernel();

		/** constructor
		 *
		 * compute custom kernel from given kernel matrix
		 * @param k kernel matrix
		 */
		CCustomKernel(CKernel* k);

		/** constructor
		 *
		 * sets full kernel matrix from full kernel matrix
		 * (from double precision floats)
		 *
		 * @param km kernel matrix
		 */
		CCustomKernel(SGMatrix<float64_t> km);

		/**
		 *
		 */
		virtual ~CCustomKernel();

		/** initialize kernel with dummy features
		 *
		 * Kernels always need feature objects assigned. As the custom kernel
		 * does not really require this it creates some magic dummy features
		 * that only know about the number of vectors
		 *
		 * @param rows features of left-hand side
		 * @param cols features of right-hand side
		 * @return if initializing was successful
		 */
		virtual bool dummy_init(int32_t rows, int32_t cols);

		/** initialize kernel
		 *
		 * @param l features of left-hand side
		 * @param r features of right-hand side
		 * @return if initializing was successful
		 */
		virtual bool init(CFeatures* l, CFeatures* r);

		/** clean up kernel */
		virtual void cleanup();

		/** return what type of kernel we are
		 *
		 * @return kernel type CUSTOM
		 */
		inline virtual EKernelType get_kernel_type() { return K_CUSTOM; }

		/** return feature type the kernel can deal with
		 *
		 * @return feature type ANY
		 */
		inline virtual EFeatureType get_feature_type() { return F_ANY; }

		/** return feature class the kernel can deal with
		 *
		 * @return feature class ANY
		 */
		inline virtual EFeatureClass get_feature_class() { return C_ANY; }

		/** return the kernel's name
		 *
		 * @return name Custom
		 */
		virtual const char* get_name() const { return "CustomKernel"; }

		/** set kernel matrix (only elements from upper triangle)
		 * from elements of upper triangle (concat'd), including the
		 * main diagonal
		 *
		 * small variant for floats64's, triangle needs to have less than 2**32 elements
		 *
		 * @param km kernel matrix
		 * @param len denotes the size of the array and should match len=cols*(cols+1)/2
		 * @return if setting was successful
		 */
		bool set_triangle_kernel_matrix_from_triangle(
			SGVector<float64_t> tri_kernel_matrix)
		{
			return set_triangle_kernel_matrix_from_triangle_generic(tri_kernel_matrix);
		}

		/** set kernel matrix (only elements from upper triangle)
		 * from elements of upper triangle (concat'd), including the
		 * main diagonal
		 *
		 * big variant, allowing the triangle to have more than 2**31-1 elements
		 *
		 * @param km kernel matrix
		 * @param len denotes the size of the array and should match len=cols*(cols+1)/2
		 * @return if setting was successful
		 */
		template <class T>
		bool set_triangle_kernel_matrix_from_triangle_generic(
			SGVector<T> tri_kernel_matrix)
		{
			ASSERT(tri_kernel_matrix.vector);

			int64_t len = tri_kernel_matrix.vlen;
			int64_t cols = (int64_t) floor(-0.5 + CMath::sqrt(0.25+2*len));

			if (cols*(cols+1)/2 != len)
			{
				SG_ERROR("km should be a vector containing a lower triangle matrix, with len=cols*(cols+1)/2 elements\n");
				return false;
			}

			cleanup_custom();
			SG_DEBUG( "using custom kernel of size %dx%d\n", cols,cols);

			kmatrix.matrix = new float32_t[len];
			kmatrix.num_rows=cols;
			kmatrix.num_cols=cols;
			upper_diagonal=true;

			for (int64_t i=0; i<len; i++)
				kmatrix.matrix[i]=tri_kernel_matrix.vector[i];

			dummy_init(cols,cols);
			return true;
		}

		/** set kernel matrix (only elements from upper triangle)
		 * from squared matrix
		 *
		 * for float64's
		 *
		 * @return if setting was successful
		 */
		inline bool set_triangle_kernel_matrix_from_full(
			SGMatrix<float64_t> full_kernel_matrix)
		{
			return set_triangle_kernel_matrix_from_full_generic(full_kernel_matrix);
		}

		/** set kernel matrix (only elements from upper triangle)
		 * from squared matrix
		 *
		 * @return if setting was successful
		 */
		template <class T>
		bool set_triangle_kernel_matrix_from_full_generic(
			SGMatrix<T> full_kernel_matrix)
		{
			int32_t rows = full_kernel_matrix.num_rows;
			int32_t cols = full_kernel_matrix.num_cols;
			ASSERT(rows==cols);

			cleanup_custom();
			SG_DEBUG( "using custom kernel of size %dx%d\n", cols,cols);

			kmatrix.matrix = new float32_t[int64_t(rows)*cols];
			kmatrix.num_rows = rows;
			kmatrix.num_cols = cols;
			upper_diagonal = false;

			for (int64_t row=0; row<rows; row++)
			{
				for (int64_t col=row; col<cols; col++)
				{
					int64_t idx=row * cols - row*(row+1)/2 + col;
					kmatrix.matrix[idx] = full_kernel_matrix.matrix[col*rows+row];
				}
			}

			dummy_init(rows, cols);
			return true;
		}

		/** set full kernel matrix from full kernel matrix
		 *
		 * for float32
		 *
		 * @return if setting was successful
		 */
		bool set_full_kernel_matrix_from_full(
			SGMatrix<float32_t> full_kernel_matrix)
		{
			cleanup_custom();
			kmatrix.matrix = full_kernel_matrix.matrix;
			kmatrix.num_rows=full_kernel_matrix.num_rows;
			kmatrix.num_cols=full_kernel_matrix.num_cols;
			dummy_init(kmatrix.num_rows, kmatrix.num_cols);
			return true;
		}

		/** set full kernel matrix from full kernel matrix
		 *
		 * for float64
		 *
		 * @return if setting was successful
		 */
		bool set_full_kernel_matrix_from_full(
			SGMatrix<float64_t> full_kernel_matrix)
		{
			cleanup_custom();
			int32_t rows=full_kernel_matrix.num_rows;
			int32_t cols=full_kernel_matrix.num_cols;
			SG_DEBUG( "using custom kernel of size %dx%d\n", rows,cols);

			kmatrix.matrix = new float32_t[int64_t(rows)*cols];
			kmatrix.num_rows = rows;
			kmatrix.num_cols = cols;
			upper_diagonal = false;

			for (int32_t row=0; row<rows; row++)
			{
				for (int32_t col=0; col<cols; col++)
					kmatrix.matrix[int64_t(row) * cols + col] =
							full_kernel_matrix.matrix[int64_t(col)*rows+row];
			}

			dummy_init(rows, cols);
			return true;
		}

		/** get number of vectors of lhs features
		 *
		 * @return number of vectors of left-hand side
		 */
		virtual inline int32_t get_num_vec_lhs()
		{
			return kmatrix.num_rows;
		}

		/** get number of vectors of rhs features
		 *
		 * @return number of vectors of right-hand side
		 */
		virtual inline int32_t get_num_vec_rhs()
		{
			return kmatrix.num_cols;
		}

		/** test whether features have been assigned to lhs and rhs
		 *
		 * @return true if features are assigned
		 */
		virtual inline bool has_features()
		{
			return (kmatrix.num_rows>0) && (kmatrix.num_cols>0);
		}

	protected:

		/** compute kernel function
		 *
		 * @param row row
		 * @param col col
		 * @return computed kernel function
		 */
		inline virtual float64_t compute(int32_t row, int32_t col)
		{
			ASSERT(kmatrix.matrix);

			if (upper_diagonal)
			{
				if (row <= col)
				{
					int64_t r=row;
					return kmatrix.matrix[r*kmatrix.num_rows - r*(r+1)/2 + col];
				}
				else
				{
					int64_t c=col;
					return kmatrix.matrix[c*kmatrix.num_cols - c*(c+1)/2 + row];
				}
			}
			else
			{
				int64_t r=row;
				return kmatrix.matrix[r*kmatrix.num_cols+col];
			}
		}

	private:

		/** only cleanup stuff specific to Custom kernel */
		void cleanup_custom();

	protected:

		/** kernel matrix */
		SGMatrix<float32_t> kmatrix;

		/** upper diagonal */
		bool upper_diagonal;
};

}
#endif /* _CUSTOMKERNEL_H__ */
