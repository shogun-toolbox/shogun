/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 1999-2009 Soeren Sonnenburg
 * Written (W) 2012 Heiko Strathmann
 * Copyright (C) 1999-2009 Fraunhofer Institute FIRST and Max-Planck-Society
 */

#ifndef _CUSTOMKERNEL_H___
#define _CUSTOMKERNEL_H___

#include <mathematics/Math.h>
#include <lib/common.h>
#include <kernel/Kernel.h>
#include <features/Features.h>

namespace shogun
{
/** @brief The Custom Kernel allows for custom user provided kernel matrices.
 *
 * For squared training matrices it allows to store only the upper triangle of
 * the kernel to save memory: Full symmetric kernel matrices can be stored as
 * is or can be internally converted into (or directly given in) upper triangle
 * representation. Also note that values are stored as 32bit floats.
 *
 * The custom kernel supports subsets each on the rows and the columns.
 *
 *
 */
class CCustomKernel: public CKernel
{
	void init();

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

		/** constructor
		 *
		 * sets full kernel matrix from full kernel matrix
		 * (from double precision floats)
		 *
		 * @param km kernel matrix
		 */
		CCustomKernel(SGMatrix<float32_t> km);

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
		 * removes subset before
		 *
		 * @param rows features of left-hand side
		 * @param cols features of right-hand side
		 * @return if initializing was successful
		 */
		virtual bool dummy_init(int32_t rows, int32_t cols);

		/** initialize kernel
		 *
		 * removes subset before
		 *
		 * @param l features of left-hand side
		 * @param r features of right-hand side
		 * @return if initializing was successful
		 */
		virtual bool init(CFeatures* l, CFeatures* r);

		/** clean up kernel */
		virtual void cleanup();

		/** only cleanup stuff specific to Custom kernel */
		void cleanup_custom();

		/** return what type of kernel we are
		 *
		 * @return kernel type CUSTOM
		 */
		virtual EKernelType get_kernel_type() { return K_CUSTOM; }

		/** return feature type the kernel can deal with
		 *
		 * @return feature type ANY
		 */
		virtual EFeatureType get_feature_type() { return F_ANY; }

		/** return feature class the kernel can deal with
		 *
		 * @return feature class ANY
		 */
		virtual EFeatureClass get_feature_class() { return C_ANY; }

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
		 * works NOT with subset
		 *
		 * @param tri_kernel_matrix tri kernel matrix
		 * @return if setting was successful
		 */
		bool set_triangle_kernel_matrix_from_triangle(
			SGVector<float64_t> tri_kernel_matrix)
		{
			if (m_row_subset_stack->has_subsets() || m_col_subset_stack->has_subsets())
			{
				SG_ERROR("%s::set_triangle_kernel_matrix_from_triangle not"
						" possible with subset. Remove first\n", get_name());
			}
			return set_triangle_kernel_matrix_from_triangle_generic(tri_kernel_matrix);
		}

		/** set kernel matrix (only elements from upper triangle)
		 * from elements of upper triangle (concat'd), including the
		 * main diagonal
		 *
		 * big variant, allowing the triangle to have more than 2**31-1 elements
		 *
		 * works NOT with subset
		 *
		 * @param tri_kernel_matrix tri kernel matrix
		 * @return if setting was successful
		 */
		template <class T>
		bool set_triangle_kernel_matrix_from_triangle_generic(
			SGVector<T> tri_kernel_matrix)
		{
			if (m_row_subset_stack->has_subsets() || m_col_subset_stack->has_subsets())
			{
				SG_ERROR("%s::set_triangle_kernel_matrix_from_triangle_generic "
						"not possible with subset. Remove first\n", get_name());
			}
			ASSERT(tri_kernel_matrix.vector)

			int64_t len = tri_kernel_matrix.vlen;
			int64_t cols = (int64_t) floor(-0.5 + CMath::sqrt(0.25+2*len));

			if (cols*(cols+1)/2 != len)
			{
				SG_ERROR("km should be a vector containing a lower triangle matrix, with len=cols*(cols+1)/2 elements\n")
				return false;
			}

			cleanup_custom();
			SG_DEBUG("using custom kernel of size %dx%d\n", cols,cols)

			kmatrix=SGMatrix<float32_t>(SG_MALLOC(float32_t, len), cols, cols);
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
		 * works NOT with subset
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
		 * works NOT with subset
		 *
		 * @return if setting was successful
		 */
		template <class T>
		bool set_triangle_kernel_matrix_from_full_generic(
			SGMatrix<T> full_kernel_matrix)
		{
			if (m_row_subset_stack->has_subsets() || m_col_subset_stack->has_subsets())
			{
				SG_ERROR("%s::set_triangle_kernel_matrix_from_full_generic "
						"not possible with subset. Remove first\n", get_name());
			}

			int32_t rows = full_kernel_matrix.num_rows;
			int32_t cols = full_kernel_matrix.num_cols;
			ASSERT(rows==cols)

			cleanup_custom();
			SG_DEBUG("using custom kernel of size %dx%d\n", cols,cols)

			kmatrix=SGMatrix<float32_t>(SG_MALLOC(float32_t, cols*(cols+1)/2), rows, cols);
			upper_diagonal = true;

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
		 * works NOT with subset
		 *
		 * @return if setting was successful
		 */
		bool set_full_kernel_matrix_from_full(
			SGMatrix<float32_t> full_kernel_matrix)
		{
			if (m_row_subset_stack->has_subsets() || m_col_subset_stack->has_subsets())
			{
				SG_ERROR("%s::set_full_kernel_matrix_from_full "
						"not possible with subset. Remove first\n", get_name());
			}

			cleanup_custom();
			kmatrix=full_kernel_matrix;
			dummy_init(kmatrix.num_rows, kmatrix.num_cols);
			return true;
		}

		/** set full kernel matrix from full kernel matrix
		 *
		 * for float64
		 *
		 * works NOT with subset
		 *
		 * @return if setting was successful
		 */
		bool set_full_kernel_matrix_from_full(
			SGMatrix<float64_t> full_kernel_matrix)
		{
			if (m_row_subset_stack->has_subsets() || m_col_subset_stack->has_subsets())
			{
				SG_ERROR("%s::set_full_kernel_matrix_from_full "
						"not possible with subset. Remove first\n", get_name());
			}

			cleanup_custom();
			int32_t rows=full_kernel_matrix.num_rows;
			int32_t cols=full_kernel_matrix.num_cols;
			SG_DEBUG("using custom kernel of size %dx%d\n", rows,cols)

			kmatrix=SGMatrix<float32_t>(rows,cols);
			upper_diagonal = false;

			for (int64_t i=0; i<int64_t(rows) * cols; i++)
				kmatrix.matrix[i]=full_kernel_matrix.matrix[i];

			dummy_init(kmatrix.num_rows, kmatrix.num_cols);
			return true;
		}

		/** adds a row subset of indices on top of the current subsets (possibly
		 * subset o subset. Calls subset_changed_post() afterwards
		 *
		 * @param subset subset of indices to add
		 * */
		virtual void add_row_subset(SGVector<index_t> subset);

		/** removes that last added row subset from subset stack, if existing
		 * Calls subset_changed_post() afterwards */
		virtual void remove_row_subset();

		/** removes all row subsets
		 * Calls subset_changed_post() afterwards */
		virtual void remove_all_row_subsets();

		/** method may be overwritten to update things that depend on subset */
		virtual void row_subset_changed_post();

		/** adds a col subset of indices on top of the current subsets (possibly
		 * subset o subset. Calls subset_changed_post() afterwards
		 *
		 * @param subset subset of indices to add
		 * */
		virtual void add_col_subset(SGVector<index_t> subset);

		/** removes that last added col subset from subset stack, if existing
		 * Calls subset_changed_post() afterwards */
		virtual void remove_col_subset();

		/** removes all col subsets
		 * Calls subset_changed_post() afterwards */
		virtual void remove_all_col_subsets();

		/** method may be overwritten to update things that depend on subset */
		virtual void col_subset_changed_post();

		/** get number of vectors of lhs features
		 *
		 * works with subset
		 *
		 * @return number of vectors of left-hand side
		 */
		virtual int32_t get_num_vec_lhs()
		{
			return m_row_subset_stack->has_subsets()
					? m_row_subset_stack->get_size() : num_lhs;
		}

		/** get number of vectors of rhs features
		 *
		 * works with subset
		 *
		 * @return number of vectors of right-hand side
		 */
		virtual int32_t get_num_vec_rhs()
		{
			return m_col_subset_stack->has_subsets()
					? m_col_subset_stack->get_size() : num_rhs;
		}

		/** test whether features have been assigned to lhs and rhs
		 *
		 * works with subset
		 *
		 * @return true if features are assigned
		 */
		virtual bool has_features()
		{
			return (get_num_vec_lhs()>0) && (get_num_vec_rhs()>0);
		}

		/** returns kernel matrix as is (not possible with subset)
		 *
		 * @return kernel matrix
		 */
		SGMatrix<float32_t> get_float32_kernel_matrix()
		{
			REQUIRE(!m_row_subset_stack, "%s::get_float32_kernel_matrix(): "
						"Not possible with row subset active! If you want to"
						" create a %s from another one with a subset, use "
						"get_kernel_matrix() and the SGMatrix constructor!\n",
						get_name(), get_name());

			REQUIRE(!m_col_subset_stack, "%s::get_float32_kernel_matrix(): "
					"Not possible with collumn subset active! If you want to"
					" create a %s from another one with a subset, use "
					"get_kernel_matrix() and the SGMatrix constructor!\n",
					get_name(), get_name());

			return kmatrix;
		}

	protected:

		/** compute kernel function
		 *
		 * works with subset
		 *
		 * @param row row
		 * @param col col
		 * @return computed kernel function
		 */
		virtual float64_t compute(int32_t row, int32_t col)
		{
			REQUIRE(kmatrix.matrix, "%s::compute(%d, %d): No kenrel matrix "
					"set!\n", get_name(), row, col);

			index_t real_row=m_row_subset_stack->subset_idx_conversion(row);
			index_t real_col=m_col_subset_stack->subset_idx_conversion(col);

			if (upper_diagonal)
			{
				if (real_row <= real_col)
				{
					int64_t r=real_row;
					return kmatrix.matrix[r*kmatrix.num_rows - r*(r+1)/2 + real_col];
				}
				else
				{
					int64_t c=real_col;
					return kmatrix.matrix[c*kmatrix.num_cols - c*(c+1)/2 + real_row];
				}
			}
			else
				return kmatrix(real_row, real_col);
		}

	protected:

		/** kernel matrix */
		SGMatrix<float32_t> kmatrix;

		/** upper diagonal */
		bool upper_diagonal;

		/** row subset stack */
		CSubsetStack* m_row_subset_stack;
		/** column subset stack */
		CSubsetStack* m_col_subset_stack;

		/** indicates whether kernel matrix is to be freed in destructor */
		bool m_free_km;
};

}
#endif /* _CUSTOMKERNEL_H__ */
