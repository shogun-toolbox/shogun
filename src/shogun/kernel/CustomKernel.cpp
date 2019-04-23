/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Heiko Strathmann, Soeren Sonnenburg, Soumyajit De, Pan Deng,
 *          Sergey Lisitsyn, Khaled Nasr, Evgeniy Andreev, Evan Shelhamer,
 *          Liang Pang
 */

#include <shogun/lib/common.h>
#include <shogun/kernel/CustomKernel.h>
#include <shogun/features/Features.h>
#include <shogun/features/DummyFeatures.h>
#include <shogun/features/IndexFeatures.h>
#include <shogun/io/SGIO.h>
#include <shogun/mathematics/linalg/LinalgNamespace.h>

using namespace shogun;
using namespace linalg;

void CustomKernel::init()
{
	m_row_subset_stack=std::make_shared<SubsetStack>();

	m_col_subset_stack=std::make_shared<SubsetStack>();

	m_is_symmetric=false;
	m_free_km=true;

	SG_ADD((std::shared_ptr<SGObject>*)&m_row_subset_stack, "row_subset_stack",
			"Subset stack of rows");
	SG_ADD((std::shared_ptr<SGObject>*)&m_col_subset_stack, "col_subset_stack",
			"Subset stack of columns");
	SG_ADD(&m_free_km, "free_km", "Whether kernel matrix should be freed in "
			"destructor");
	SG_ADD(&m_is_symmetric, "is_symmetric", "Whether kernel matrix is symmetric");
	SG_ADD(&kmatrix, "kmatrix", "Kernel matrix.");
	SG_ADD(&upper_diagonal, "upper_diagonal", "Upper diagonal");
}

CustomKernel::CustomKernel()
: Kernel(10), kmatrix(), upper_diagonal(false)
{
	SG_TRACE("created CustomKernel");
	init();
}

CustomKernel::CustomKernel(std::shared_ptr<Kernel> k)
: Kernel(10)
{
	SG_TRACE("created CustomKernel");
	init();

	/* if constructed from a custom kernel, use same kernel matrix */
	if (k->get_kernel_type()==K_CUSTOM)
	{
		auto casted=std::static_pointer_cast<CustomKernel>(k);
		m_is_symmetric=casted->m_is_symmetric;
		set_full_kernel_matrix_from_full(casted->get_float32_kernel_matrix());
		m_free_km=false;
	}
	else
	{
		m_is_symmetric=k->get_lhs_equals_rhs();
		set_full_kernel_matrix_from_full(k->get_kernel_matrix());
	}
}

CustomKernel::CustomKernel(SGMatrix<float64_t> km)
: Kernel(10), upper_diagonal(false)
{
	SG_TRACE("Entering");
	init();
	set_full_kernel_matrix_from_full(km, true);
	SG_TRACE("Leaving");
}

CustomKernel::CustomKernel(SGMatrix<float32_t> km)
: Kernel(10), upper_diagonal(false)
{
	SG_TRACE("Entering");
	init();
	set_full_kernel_matrix_from_full(km, true);
	SG_TRACE("Leaving");
}

CustomKernel::~CustomKernel()
{
	SG_TRACE("Entering");
	cleanup();
	SG_TRACE("Leaving");
}

bool CustomKernel::dummy_init(int32_t rows, int32_t cols)
{
	return init(std::make_shared<DummyFeatures>(rows), std::make_shared<DummyFeatures>(cols));
}

bool CustomKernel::init(std::shared_ptr<Features> l, std::shared_ptr<Features> r)
{
	/* make it possible to call with NULL values since features are useless
	 * for custom kernel matrix */
	if (!l)
		l=lhs;

	if (!r)
		r=rhs;

	/* Make sure l and r should not be NULL */
	require(l, "Features l should not be NULL");
	require(r, "Features r should not be NULL");

	/* Make sure l and r have the same type of CFeatures */
	require(l->get_feature_class()==r->get_feature_class(),
			"Different FeatureClass: l is {}, r is {}",
			l->get_feature_class(),r->get_feature_class());
	require(l->get_feature_type()==r->get_feature_type(),
			"Different FeatureType: l is {}, r is {}",
			l->get_feature_type(),r->get_feature_type());

	/* If l and r are the type of IndexFeatures,
	 * the init function adds a subset to kernel matrix.
	 * Then call get_kernel_matrix will get the submatrix
	 * of the kernel matrix.
	 */
	if (l->get_feature_class()==C_INDEX && r->get_feature_class()==C_INDEX)
	{
		auto l_idx = std::static_pointer_cast<IndexFeatures>(l);
		auto r_idx = std::static_pointer_cast<IndexFeatures>(r);

		remove_all_col_subsets();
		remove_all_row_subsets();

		add_row_subset(l_idx->get_feature_index());
		add_col_subset(r_idx->get_feature_index());

		lhs_equals_rhs=m_is_symmetric;

		return true;
	}

	/* For other types of Features do the default actions below */
	Kernel::init(l, r);

	lhs_equals_rhs=m_is_symmetric;

	SG_DEBUG("num_vec_lhs: {} vs num_rows {}", l->get_num_vectors(), kmatrix.num_rows)
	SG_DEBUG("num_vec_rhs: {} vs num_cols {}", r->get_num_vectors(), kmatrix.num_cols)
	ASSERT(l->get_num_vectors()==kmatrix.num_rows)
	ASSERT(r->get_num_vectors()==kmatrix.num_cols)
	return init_normalizer();
}

float64_t CustomKernel::sum_symmetric_block(index_t block_begin,
		index_t block_size, bool no_diag)
{
	SG_TRACE("Entering");

	if (m_row_subset_stack->has_subsets() || m_col_subset_stack->has_subsets())
	{
		io::info("Row/col subsets initialized! Falling back to "
				"Kernel::sum_symmetric_block (slower)!");
		return Kernel::sum_symmetric_block(block_begin, block_size, no_diag);
	}

	require(kmatrix.matrix, "The kernel matrix is not initialized!");
	require(m_is_symmetric, "The kernel matrix is not symmetric!");
	require(block_begin>=0 && block_begin<kmatrix.num_cols,
			"Invalid block begin index ({}, {})!", block_begin, block_begin);
	require(block_begin+block_size<=kmatrix.num_cols,
			"Invalid block size ({}) at starting index ({}, {})! "
			"Please use smaller blocks!", block_size, block_begin, block_begin);
	require(block_size>=1, "Invalid block size ({})!", block_size);

	SG_TRACE("Leaving");

	return sum_symmetric(block(kmatrix, block_begin,
				block_begin, block_size, block_size), no_diag);
}

float64_t CustomKernel::sum_block(index_t block_begin_row,
		index_t block_begin_col, index_t block_size_row,
		index_t block_size_col, bool no_diag)
{
	SG_TRACE("Entering");

	if (m_row_subset_stack->has_subsets() || m_col_subset_stack->has_subsets())
	{
		io::info("Row/col subsets initialized! Falling back to "
				"Kernel::sum_block (slower)!");
		return Kernel::sum_block(block_begin_row, block_begin_col,
				block_size_row, block_size_col, no_diag);
	}

	require(kmatrix.matrix, "The kernel matrix is not initialized!");
	require(block_begin_row>=0 && block_begin_row<kmatrix.num_rows &&
			block_begin_col>=0 && block_begin_col<kmatrix.num_cols,
			"Invalid block begin index ({}, {})!",
			block_begin_row, block_begin_col);
	require(block_begin_row+block_size_row<=kmatrix.num_rows &&
			block_begin_col+block_size_col<=kmatrix.num_cols,
			"Invalid block size ({}, {}) at starting index ({}, {})! "
			"Please use smaller blocks!", block_size_row, block_size_col,
			block_begin_row, block_begin_col);
	require(block_size_row>=1 && block_size_col>=1,
			"Invalid block size ({}, {})!", block_size_row, block_size_col);

	// check if removal of diagonal is required/valid
	if (no_diag && block_size_row!=block_size_col)
	{
		io::warn("Not removing the main diagonal since block is not square!");
		no_diag=false;
	}

	SG_TRACE("Leaving");

	return sum(block(kmatrix, block_begin_row, block_begin_col,
				block_size_row, block_size_col), no_diag);
}

SGVector<float64_t> CustomKernel::row_wise_sum_symmetric_block(index_t
		block_begin, index_t block_size, bool no_diag)
{
	SG_TRACE("Entering");

	if (m_row_subset_stack->has_subsets() || m_col_subset_stack->has_subsets())
	{
		io::info("Row/col subsets initialized! Falling back to "
				"Kernel::row_wise_sum_symmetric_block (slower)!");
		return Kernel::row_wise_sum_symmetric_block(block_begin, block_size,
				no_diag);
	}

	require(kmatrix.matrix, "The kernel matrix is not initialized!");
	require(m_is_symmetric, "The kernel matrix is not symmetric!");
	require(block_begin>=0 && block_begin<kmatrix.num_cols,
			"Invalid block begin index ({}, {})!", block_begin, block_begin);
	require(block_begin+block_size<=kmatrix.num_cols,
			"Invalid block size ({}) at starting index ({}, {})! "
			"Please use smaller blocks!", block_size, block_begin, block_begin);
	require(block_size>=1, "Invalid block size ({})!", block_size);

	SGVector<float32_t> s=rowwise_sum(block(kmatrix, block_begin,
				block_begin, block_size, block_size), no_diag);

	// casting to float64_t vector
	SGVector<float64_t> sum(s.vlen);
	for (index_t i=0; i<s.vlen; ++i)
		sum[i]=s[i];

	SG_TRACE("Leaving");

	return sum;
}

SGMatrix<float64_t> CustomKernel::row_wise_sum_squared_sum_symmetric_block(
		index_t block_begin, index_t block_size, bool no_diag)
{
	SG_TRACE("Entering");

	if (m_row_subset_stack->has_subsets() || m_col_subset_stack->has_subsets())
	{
		io::info("Row/col subsets initialized! Falling back to "
				"Kernel::row_wise_sum_squared_sum_symmetric_block (slower)!");
		return Kernel::row_wise_sum_squared_sum_symmetric_block(block_begin,
				block_size, no_diag);
	}

	require(kmatrix.matrix, "The kernel matrix is not initialized!");
	require(m_is_symmetric, "The kernel matrix is not symmetric!");
	require(block_begin>=0 && block_begin<kmatrix.num_cols,
			"Invalid block begin index ({}, {})!", block_begin, block_begin);
	require(block_begin+block_size<=kmatrix.num_cols,
			"Invalid block size ({}) at starting index ({}, {})! "
			"Please use smaller blocks!", block_size, block_begin, block_begin);
	require(block_size>=1, "Invalid block size ({})!", block_size);

	// initialize the matrix that accumulates the row/col-wise sum
	// the first column stores the sum of kernel values
	// the second column stores the sum of squared kernel values
	SGMatrix<float64_t> row_sum(block_size, 2);

	SGVector<float32_t> sum=rowwise_sum(block(kmatrix,
				block_begin, block_begin, block_size, block_size), no_diag);

	auto kmatrix_block = block(kmatrix, block_begin, block_begin, block_size, block_size);
	SGVector<float32_t> sq_sum=rowwise_sum(
		element_prod(kmatrix_block, kmatrix_block), no_diag);

	for (index_t i=0; i<sum.vlen; ++i)
		row_sum(i, 0)=sum[i];

	for (index_t i=0; i<sq_sum.vlen; ++i)
		row_sum(i, 1)=sq_sum[i];

	SG_TRACE("Leaving");

	return row_sum;
}

SGVector<float64_t> CustomKernel::row_col_wise_sum_block(index_t
		block_begin_row, index_t block_begin_col, index_t block_size_row,
		index_t block_size_col, bool no_diag)
{
	SG_TRACE("Entering");

	if (m_row_subset_stack->has_subsets() || m_col_subset_stack->has_subsets())
	{
		io::info("Row/col subsets initialized! Falling back to "
				"Kernel::row_col_wise_sum_block (slower)!");
		return Kernel::row_col_wise_sum_block(block_begin_row, block_begin_col,
				block_size_row, block_size_col, no_diag);
	}

	require(kmatrix.matrix, "The kernel matrix is not initialized!");
	require(block_begin_row>=0 && block_begin_row<kmatrix.num_rows &&
			block_begin_col>=0 && block_begin_col<kmatrix.num_cols,
			"Invalid block begin index ({}, {})!",
			block_begin_row, block_begin_col);
	require(block_begin_row+block_size_row<=kmatrix.num_rows &&
			block_begin_col+block_size_col<=kmatrix.num_cols,
			"Invalid block size ({}, {}) at starting index ({}, {})! "
			"Please use smaller blocks!", block_size_row, block_size_col,
			block_begin_row, block_begin_col);
	require(block_size_row>=1 && block_size_col>=1,
			"Invalid block size ({}, {})!", block_size_row, block_size_col);

	// check if removal of diagonal is required/valid
	if (no_diag && block_size_row!=block_size_col)
	{
		io::warn("Not removing the main diagonal since block is not square!");
		no_diag=false;
	}

	// initialize the vector that accumulates the row/col-wise sum
	// the first block_size_row entries store the row-wise sum of kernel values
	// the nextt block_size_col entries store the col-wise sum of kernel values
	SGVector<float64_t> sum(block_size_row+block_size_col);

	SGVector<float32_t> rowwise=rowwise_sum(block(kmatrix,
				block_begin_row, block_begin_col, block_size_row,
				block_size_col), no_diag);

	SGVector<float32_t> colwise=colwise_sum(block(kmatrix,
				block_begin_row, block_begin_col, block_size_row,
				block_size_col), no_diag);

	for (index_t i=0; i<rowwise.vlen; ++i)
		sum[i]=rowwise[i];

	for (index_t i=0; i<colwise.vlen; ++i)
		sum[i+rowwise.vlen]=colwise[i];

	SG_TRACE("Leaving");

	return sum;
}

void CustomKernel::cleanup_custom()
{
	SG_TRACE("Entering");
	remove_all_row_subsets();
	remove_all_col_subsets();

	kmatrix=SGMatrix<float32_t>();
	upper_diagonal=false;

	SG_TRACE("Leaving");
}

void CustomKernel::cleanup()
{
	cleanup_custom();
	Kernel::cleanup();
}

void CustomKernel::add_row_subset(SGVector<index_t> subset)
{
	m_row_subset_stack->add_subset(subset);
	row_subset_changed_post();
}

void CustomKernel::add_row_subset_in_place(SGVector<index_t> subset)
{
	m_row_subset_stack->add_subset_in_place(subset);
	row_subset_changed_post();
}

void CustomKernel::remove_row_subset()
{
	m_row_subset_stack->remove_subset();
	row_subset_changed_post();
}

void CustomKernel::remove_all_row_subsets()
{
	m_row_subset_stack->remove_all_subsets();
	row_subset_changed_post();
}

void CustomKernel::row_subset_changed_post()
{
	if (m_row_subset_stack->has_subsets())
		num_lhs=m_row_subset_stack->get_size();
	else
		num_lhs=kmatrix.num_rows;
}

void CustomKernel::add_col_subset(SGVector<index_t> subset)
{
	m_col_subset_stack->add_subset(subset);
	col_subset_changed_post();
}

void CustomKernel::add_col_subset_in_place(SGVector<index_t> subset)
{
	m_col_subset_stack->add_subset_in_place(subset);
	col_subset_changed_post();
}

void CustomKernel::remove_col_subset()
{
	m_col_subset_stack->remove_subset();
	col_subset_changed_post();
}

void CustomKernel::remove_all_col_subsets()
{
	m_col_subset_stack->remove_all_subsets();
	col_subset_changed_post();
}

void CustomKernel::col_subset_changed_post()
{
	if (m_col_subset_stack->has_subsets())
		num_rhs=m_col_subset_stack->get_size();
	else
		num_rhs=kmatrix.num_cols;
}
