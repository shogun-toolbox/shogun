/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Heiko Strathmann, Soeren Sonnenburg, Soumyajit De, Evgeniy Andreev,
 *          Sergey Lisitsyn, Yuyu Zhang, Evan Shelhamer, Pan Deng
 */

#ifndef _CUSTOMKERNEL_H___
#define _CUSTOMKERNEL_H___

#include <shogun/lib/config.h>

#include <shogun/mathematics/Math.h>
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
 * The custom kernel supports subsets each on the rows and the columns. See
 * documentation in Features, Labels how this works. The interface is similar.
 *
 *
 */
class CustomKernel: public Kernel
{
	void init();

	public:
		/** default constructor */
		CustomKernel();

		/** constructor
		 *
		 * compute custom kernel from given kernel matrix
		 * @param k kernel matrix
		 */
		CustomKernel(const std::shared_ptr<Kernel>& k);

		/** constructor
		 *
		 * sets full kernel matrix from full kernel matrix
		 * (from double precision floats)
		 *
		 * @param km kernel matrix
		 */
		CustomKernel(SGMatrix<float64_t> km);

		/** constructor
		 *
		 * sets full kernel matrix from full kernel matrix
		 * (from double precision floats)
		 *
		 * @param km kernel matrix
		 */
		CustomKernel(SGMatrix<float32_t> km);

		/**
		 *
		 */
		~CustomKernel() override;

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
		bool init(std::shared_ptr<Features> l, std::shared_ptr<Features> r) override;

		/** clean up kernel */
		void cleanup() override;

		/** only cleanup stuff specific to Custom kernel */
		void cleanup_custom();

		/** return what type of kernel we are
		 *
		 * @return kernel type CUSTOM
		 */
		EKernelType get_kernel_type() override { return K_CUSTOM; }

		/** return feature type the kernel can deal with
		 *
		 * @return feature type ANY
		 */
		EFeatureType get_feature_type() override { return F_ANY; }

		/** return feature class the kernel can deal with
		 *
		 * @return feature class ANY
		 */
		EFeatureClass get_feature_class() override { return C_ANY; }

		/** return the kernel's name
		 *
		 * @return name Custom
		 */
		const char* get_name() const override { return "CustomKernel"; }

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
				error("{}::set_triangle_kernel_matrix_from_triangle not"
						" possible with subset. Remove first", get_name());
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
				error("{}::set_triangle_kernel_matrix_from_triangle_generic "
						"not possible with subset. Remove first", get_name());
			}
			ASSERT(tri_kernel_matrix.vector)

			int64_t len = tri_kernel_matrix.vlen;
			int64_t cols = (int64_t)floor(-0.5 + std::sqrt(0.25 + 2 * len));

			if (cols*(cols+1)/2 != len)
			{
				error("km should be a vector containing a lower triangle matrix, with len=cols*(cols+1)/2 elements");
				return false;
			}

			cleanup_custom();
			SG_DEBUG("using custom kernel of size {}x{}", cols,cols)

			float32_t* m = SG_MALLOC(float32_t, len);
			kmatrix=SGMatrix<float32_t>(m, cols, cols);
			upper_diagonal=true;

			for (int64_t i=0; i<len; i++)
				kmatrix.matrix[i]=tri_kernel_matrix.vector[i];

			m_is_symmetric=true;
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
				error("{}::set_triangle_kernel_matrix_from_full_generic "
						"not possible with subset. Remove first", get_name());
			}

			int32_t rows = full_kernel_matrix.num_rows;
			int32_t cols = full_kernel_matrix.num_cols;
			ASSERT(rows==cols)

			cleanup_custom();
			SG_DEBUG("using custom kernel of size {}x{}", cols,cols)

			float32_t* m = SG_MALLOC(float32_t, cols*(cols+1)/2);
			kmatrix=SGMatrix<float32_t>(m, rows, cols);
			upper_diagonal = true;

			for (int64_t row=0; row<rows; row++)
			{
				for (int64_t col=row; col<cols; col++)
				{
					int64_t idx=row * cols - row*(row+1)/2 + col;
					kmatrix.matrix[idx] = full_kernel_matrix.matrix[col*rows+row];
				}
			}

			m_is_symmetric=true;
			dummy_init(rows, cols);
			return true;
		}

		/** set full kernel matrix from full kernel matrix
		 *
		 * for float32
		 *
		 * works NOT with subset
		 *
		 * @param full_kernel_matrix the original kernel matrix to be set from
		 * @param check_symmetry whether checking for symmetry of the kernel
		 * matrix is required
		 *
		 * @return if setting was successful
		 */
		bool set_full_kernel_matrix_from_full(
			SGMatrix<float32_t> full_kernel_matrix, bool check_symmetry=false)
		{
			if (m_row_subset_stack->has_subsets() || m_col_subset_stack->has_subsets())
			{
				error("{}::set_full_kernel_matrix_from_full "
						"not possible with subset. Remove first", get_name());
			}

			cleanup_custom();
			kmatrix=full_kernel_matrix;

			if (check_symmetry)
				m_is_symmetric=kmatrix.is_symmetric();

			dummy_init(kmatrix.num_rows, kmatrix.num_cols);
			return true;
		}

		/** set full kernel matrix from full kernel matrix
		 *
		 * for float64
		 *
		 * works NOT with subset
		 *
		 * @param full_kernel_matrix the original kernel matrix to be set from
		 * @param check_symmetry whether checking for symmetry of the kernel
		 * matrix is required
		 *
		 * @return if setting was successful
		 */
		bool set_full_kernel_matrix_from_full(
			SGMatrix<float64_t> full_kernel_matrix, bool check_symmetry=false)
		{
			if (m_row_subset_stack->has_subsets() || m_col_subset_stack->has_subsets())
			{
				error("{}::set_full_kernel_matrix_from_full "
						"not possible with subset. Remove first", get_name());
			}

			cleanup_custom();
			int32_t rows=full_kernel_matrix.num_rows;
			int32_t cols=full_kernel_matrix.num_cols;
			SG_DEBUG("using custom kernel of size {}x{}", rows,cols)

			kmatrix=SGMatrix<float32_t>(rows,cols);
			upper_diagonal = false;

			for (int64_t i=0; i<int64_t(rows) * cols; i++)
				kmatrix.matrix[i]=full_kernel_matrix.matrix[i];

			if (check_symmetry)
				m_is_symmetric=kmatrix.is_symmetric();

			dummy_init(kmatrix.num_rows, kmatrix.num_cols);
			return true;
		}

		/**
		 * Overrides the sum_symmetric_block method of Kernel to compute the
		 * sum directly from the precomputed kernel matrix.
		 * (Falls back to Kernel method if subsets are specified).
		 *
		 * @param block_begin the row and col index at which the block starts
		 * @param block_size the number of rows and cols in the block
		 * @param no_diag if true (default), the diagonal elements are excluded
		 * from the sum
		 *
		 * @return sum of kernel values within the block computed as
		 * \f[
		 *	\sum_{i}\sum_{j}k(i+\text{block-begin}, j+\text{block-begin})
		 * \f]
		 * where \f$i,j\in[0,\text{block-size}-1]\f$
		 */
		float64_t sum_symmetric_block(index_t block_begin,
				index_t block_size, bool no_diag=true) override;

		/**
		 * Overrides the sum_block method of Kernel to compute the
		 * sum directly from the precomputed kernel matrix.
		 * (Falls back to Kernel method if subsets are specified).
		 *
		 * @param block_begin_row the row index at which the block starts
		 * @param block_begin_col the col index at which the block starts
		 * @param block_size_row the number of rows in the block
		 * @param block_size_col the number of cols in the block
		 * @param no_diag if true (default is false), the diagonal elements
		 * are excluded from the sum, provided that block_size_row
		 * and block_size_col are same (i.e. the block is square). Otherwise,
		 * these are always added
		 *
		 * @return sum of kernel values within the block computed as
		 * \f[
		 *	\sum_{i}\sum_{j}k(i+\text{block-begin-row}, j+\text{block-begin-col})
		 * \f]
		 * where \f$i\in[0,\text{block-size-row}-1]\f$ and
		 * \f$j\in[0,\text{block-size-col}-1]\f$
		 */
		float64_t sum_block(index_t block_begin_row,
				index_t block_begin_col, index_t block_size_row,
				index_t block_size_col, bool no_diag=false) override;

		/**
		 * Overrides the row_wise_sum_symmetric_block method of Kernel to compute the
		 * sum directly from the precomputed kernel matrix.
		 * (Falls back to Kernel method if subsets are specified).
		 *
		 * @param block_begin the row and col index at which the block starts
		 * @param block_size the number of rows and cols in the block
		 * @param no_diag if true (default), the diagonal elements are excluded
		 * from the row/col-wise sum
		 *
		 * @return vector containing row-wise sum computed as
		 * \f[
		 *	v[i]=\sum_{j}k(i+\text{block-begin}, j+\text{block-begin})
		 * \f]
		 * where \f$i,j\in[0,\text{block-size}-1]\f$
		 */
		SGVector<float64_t> row_wise_sum_symmetric_block(index_t
				block_begin, index_t block_size, bool no_diag=true) override;

		/**
		 * Overrides the row_wise_sum_squared_sum_symmetric_block method of
		 * Kernel to compute the sum directly from the precomputed kernel matrix.
		 * (Falls back to Kernel method if subsets are specified).
		 *
		 * @param block_begin the row and col index at which the block starts
		 * @param block_size the number of rows and cols in the block
		 * @param no_diag if true (default), the diagonal elements are excluded
		 * from the row/col-wise sum
		 *
		 * @return a matrix whose first column contains the row-wise sum of
		 * kernel values computed as
		 * \f[
		 *	v_0[i]=\sum_{j}k(i+\text{block-begin}, j+\text{block-begin})
		 * \f]
		 * and second column contains the row-wise sum of squared kernel values
		 * \f[
		 *	v_1[i]=\sum_{j}^k^2(i+\text{block-begin}, j+\text{block-begin})
		 * \f]
		 * where \f$i,j\in[0,\text{block-size}-1]\f$
		 */
		SGMatrix<float64_t> row_wise_sum_squared_sum_symmetric_block(
				index_t block_begin, index_t block_size, bool no_diag=true) override;

		/**
		 * Overrides the row_wise_sum_block method of Kernel to compute the sum
		 * directly from the precomputed kernel matrix.
		 * (Falls back to Kernel method if subsets are specified).
		 *
		 * @param block_begin_row the row index at which the block starts
		 * @param block_begin_col the col index at which the block starts
		 * @param block_size_row the number of rows in the block
		 * @param block_size_col the number of cols in the block
		 * @param no_diag if true (default is false), the diagonal elements
		 * are excluded from the row/col-wise sum, provided that block_size_row
		 * and block_size_col are same (i.e. the block is square). Otherwise,
		 * these are always added
		 *
		 * @return a vector whose first block_size_row entries contain
		 * row-wise sum of kernel values computed as
		 * \f[
		 *	v[i]=\sum_{j}k(i+\text{block-begin-row}, j+\text{block-begin-col})
		 * \f]
		 * and rest block_size_col entries col-wise sum of kernel values
		 * computed as
		 * \f[
		 *	v[\text{block-size-row}+j]=\sum_{i}k(i+\text{block-begin-row},
		 *	j+\text{block-begin-col})
		 * \f]
		 * where \f$i\in[0,\text{block-size-row}-1]\f$ and
		 * \f$j\in[0,\text{block-size-col}-1]\f$
		 */
		SGVector<float64_t> row_col_wise_sum_block(
				index_t block_begin_row, index_t block_begin_col,
				index_t block_size_row, index_t block_size_col,
				bool no_diag=false) override;

		/** Adds a row subset of indices on top of the current subsets (possibly
		 * subset of subset). Every call causes a new active index vector
		 * to be stored. Added subsets can be removed one-by-one. If this is not
		 * needed, add_row_subset_in_place() should be used (does not store
		 * intermediate index vectors)
		 *
		 * Calls row_subset_changed_post() afterwards
		 *
		 * @param subset subset of indices to add
		 * */
		virtual void add_row_subset(SGVector<index_t> subset);

		/** Sets/changes latest added row subset. This allows to add multiple subsets
		 * with in-place memory requirements. They cannot be removed one-by-one
		 * afterwards, only the latest active can. If this is needed, use
		 * add_row_subset(). If no subset is active, this just adds.
		 *
		 * Calls row_subset_changed_post() afterwards
		 *
		 * @param subset subset of indices to replace the latest one with.
		 * */
		virtual void add_row_subset_in_place(SGVector<index_t> subset);

		/** removes that last added row subset from subset stack, if existing
		 * Calls subset_changed_post() afterwards */
		virtual void remove_row_subset();

		/** removes all row subsets
		 * Calls subset_changed_post() afterwards */
		virtual void remove_all_row_subsets();

		/** method may be overwritten to update things that depend on subset */
		virtual void row_subset_changed_post();

		/** Adds a column subset of indices on top of the current subsets (possibly
		 * subset of subset). Every call causes a new active index vector
		 * to be stored. Added subsets can be removed one-by-one. If this is not
		 * needed, add_col_subset_in_place() should be used (does not store
		 * intermediate index vectors)
		 *
		 * Calls col_subset_changed_post() afterwards
		 *
		 * @param subset subset of indices to add
		 * */
		virtual void add_col_subset(SGVector<index_t> subset);

		/** Sets/changes latest added column subset. This allows to add multiple subsets
		 * with in-place memory requirements. They cannot be removed one-by-one
		 * afterwards, only the latest active can. If this is needed, use
		 * add_col_subset(). If no subset is active, this just adds.
		 *
		 * Calls col_subset_changed_post() afterwards
		 *
		 * @param subset subset of indices to replace the latest one with.
		 * */
		virtual void add_col_subset_in_place(SGVector<index_t> subset);

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
		int32_t get_num_vec_lhs() override
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
		int32_t get_num_vec_rhs() override
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
		bool has_features() override
		{
			return (get_num_vec_lhs()>0) && (get_num_vec_rhs()>0);
		}

		/** returns kernel matrix as is (not possible with subset)
		 *
		 * @return kernel matrix
		 */
		SGMatrix<float32_t> get_float32_kernel_matrix()
		{
			require(!m_row_subset_stack->has_subsets(), "{}::get_float32_kernel_matrix(): "
						"Not possible with row subset active! If you want to"
						" create a {} from another one with a subset, use "
						"get_kernel_matrix() and the SGMatrix constructor!",
						get_name(), get_name());

			require(!m_col_subset_stack->has_subsets(), "{}::get_float32_kernel_matrix(): "
					"Not possible with collumn subset active! If you want to"
					" create a {} from another one with a subset, use "
					"get_kernel_matrix() and the SGMatrix constructor!",
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
		float64_t compute(int32_t row, int32_t col) override
		{
			require(kmatrix.matrix, "{}::compute({}, {}): No kenrel matrix "
					"set!", get_name(), row, col);

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

		/** whether the kernel matrix is symmetric */
		bool m_is_symmetric;

		/** row subset stack */
		std::shared_ptr<SubsetStack> m_row_subset_stack;

		/** column subset stack */
		std::shared_ptr<SubsetStack> m_col_subset_stack;

		/** indicates whether kernel matrix is to be freed in destructor */
		bool m_free_km;
};

}
#endif /* _CUSTOMKERNEL_H__ */
