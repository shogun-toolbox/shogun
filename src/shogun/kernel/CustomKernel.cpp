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

#include <shogun/lib/common.h>
#include <shogun/kernel/CustomKernel.h>
#include <shogun/features/Features.h>
#include <shogun/features/DummyFeatures.h>
#include <shogun/features/IndexFeatures.h>
#include <shogun/io/SGIO.h>

using namespace shogun;

#ifdef HAVE_LINALG_LIB
#include <shogun/mathematics/linalg/linalg.h>

using namespace linalg;
#endif // HAVE_LINALG_LIB

void CCustomKernel::init()
{
	m_row_subset_stack=new CSubsetStack();
	SG_REF(m_row_subset_stack)
	m_col_subset_stack=new CSubsetStack();
	SG_REF(m_col_subset_stack)
	m_is_symmetric=false;
	m_free_km=true;

	SG_ADD((CSGObject**)&m_row_subset_stack, "row_subset_stack",
			"Subset stack of rows", MS_NOT_AVAILABLE);
	SG_ADD((CSGObject**)&m_col_subset_stack, "col_subset_stack",
			"Subset stack of columns", MS_NOT_AVAILABLE);
	SG_ADD(&m_free_km, "free_km", "Whether kernel matrix should be freed in "
			"destructor", MS_NOT_AVAILABLE);
	SG_ADD(&m_is_symmetric, "is_symmetric", "Whether kernel matrix is symmetric",
			MS_NOT_AVAILABLE);
	SG_ADD(&kmatrix, "kmatrix", "Kernel matrix.", MS_NOT_AVAILABLE);
	SG_ADD(&upper_diagonal, "upper_diagonal", "Upper diagonal", MS_NOT_AVAILABLE);
}

CCustomKernel::CCustomKernel()
: CKernel(10), kmatrix(), upper_diagonal(false)
{
	SG_DEBUG("created CCustomKernel\n")
	init();
}

CCustomKernel::CCustomKernel(CKernel* k)
: CKernel(10)
{
	SG_DEBUG("created CCustomKernel\n")
	init();

	/* if constructed from a custom kernel, use same kernel matrix */
	if (k->get_kernel_type()==K_CUSTOM)
	{
		CCustomKernel* casted=(CCustomKernel*)k;
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

CCustomKernel::CCustomKernel(SGMatrix<float64_t> km)
: CKernel(10), upper_diagonal(false)
{
	SG_DEBUG("Entering\n")
	init();
	set_full_kernel_matrix_from_full(km, true);
	SG_DEBUG("Leaving\n")
}

CCustomKernel::CCustomKernel(SGMatrix<float32_t> km)
: CKernel(10), upper_diagonal(false)
{
	SG_DEBUG("Entering\n")
	init();
	set_full_kernel_matrix_from_full(km, true);
	SG_DEBUG("Leaving\n")
}

CCustomKernel::~CCustomKernel()
{
	SG_DEBUG("Entering\n")
	cleanup();
	SG_UNREF(m_row_subset_stack);
	SG_UNREF(m_col_subset_stack);
	SG_DEBUG("Leaving\n")
}

bool CCustomKernel::dummy_init(int32_t rows, int32_t cols)
{
	return init(new CDummyFeatures(rows), new CDummyFeatures(cols));
}

bool CCustomKernel::init(CFeatures* l, CFeatures* r)
{
	/* make it possible to call with NULL values since features are useless
	 * for custom kernel matrix */
	if (!l)
		l=lhs;

	if (!r)
		r=rhs;

	/* Make sure l and r should not be NULL */
	REQUIRE(l, "CFeatures l should not be NULL\n")
	REQUIRE(r, "CFeatures r should not be NULL\n")

	/* Make sure l and r have the same type of CFeatures */
	REQUIRE(l->get_feature_class()==r->get_feature_class(),
			"Different FeatureClass: l is %d, r is %d\n",
			l->get_feature_class(),r->get_feature_class())
	REQUIRE(l->get_feature_type()==r->get_feature_type(),
			"Different FeatureType: l is %d, r is %d\n",
			l->get_feature_type(),r->get_feature_type())

	/* If l and r are the type of CIndexFeatures,
	 * the init function adds a subset to kernel matrix.
	 * Then call get_kernel_matrix will get the submatrix
	 * of the kernel matrix.
	 */
	if (l->get_feature_class()==C_INDEX && r->get_feature_class()==C_INDEX)
	{
		CIndexFeatures* l_idx = (CIndexFeatures*)l;
		CIndexFeatures* r_idx = (CIndexFeatures*)r;

		remove_all_col_subsets();
		remove_all_row_subsets();

		add_row_subset(l_idx->get_feature_index());
		add_col_subset(r_idx->get_feature_index());

		lhs_equals_rhs=m_is_symmetric;

		return true;
	}

	/* For other types of CFeatures do the default actions below */
	CKernel::init(l, r);

	lhs_equals_rhs=m_is_symmetric;

	SG_DEBUG("num_vec_lhs: %d vs num_rows %d\n", l->get_num_vectors(), kmatrix.num_rows)
	SG_DEBUG("num_vec_rhs: %d vs num_cols %d\n", r->get_num_vectors(), kmatrix.num_cols)
	ASSERT(l->get_num_vectors()==kmatrix.num_rows)
	ASSERT(r->get_num_vectors()==kmatrix.num_cols)
	return init_normalizer();
}

#ifdef HAVE_LINALG_LIB
float64_t CCustomKernel::sum_symmetric_block(index_t block_begin,
		index_t block_size, bool no_diag)
{
	SG_DEBUG("Entering\n");

	if (m_row_subset_stack->has_subsets() || m_col_subset_stack->has_subsets())
	{
		SG_INFO("Row/col subsets initialized! Falling back to "
				"CKernel::sum_symmetric_block (slower)!\n");
		return CKernel::sum_symmetric_block(block_begin, block_size, no_diag);
	}

	REQUIRE(kmatrix.matrix, "The kernel matrix is not initialized!\n")
	REQUIRE(m_is_symmetric, "The kernel matrix is not symmetric!\n")
	REQUIRE(block_begin>=0 && block_begin<kmatrix.num_cols,
			"Invalid block begin index (%d, %d)!\n", block_begin, block_begin)
	REQUIRE(block_begin+block_size<=kmatrix.num_cols,
			"Invalid block size (%d) at starting index (%d, %d)! "
			"Please use smaller blocks!", block_size, block_begin, block_begin)
	REQUIRE(block_size>=1, "Invalid block size (%d)!\n", block_size)

	SG_DEBUG("Leaving\n");

	return sum_symmetric<Backend::EIGEN3>(block(kmatrix, block_begin,
				block_begin, block_size, block_size), no_diag);
}

float64_t CCustomKernel::sum_block(index_t block_begin_row,
		index_t block_begin_col, index_t block_size_row,
		index_t block_size_col, bool no_diag)
{
	SG_DEBUG("Entering\n");

	if (m_row_subset_stack->has_subsets() || m_col_subset_stack->has_subsets())
	{
		SG_INFO("Row/col subsets initialized! Falling back to "
				"CKernel::sum_block (slower)!\n");
		return CKernel::sum_block(block_begin_row, block_begin_col,
				block_size_row, block_size_col, no_diag);
	}

	REQUIRE(kmatrix.matrix, "The kernel matrix is not initialized!\n")
	REQUIRE(block_begin_row>=0 && block_begin_row<kmatrix.num_rows &&
			block_begin_col>=0 && block_begin_col<kmatrix.num_cols,
			"Invalid block begin index (%d, %d)!\n",
			block_begin_row, block_begin_col)
	REQUIRE(block_begin_row+block_size_row<=kmatrix.num_rows &&
			block_begin_col+block_size_col<=kmatrix.num_cols,
			"Invalid block size (%d, %d) at starting index (%d, %d)! "
			"Please use smaller blocks!", block_size_row, block_size_col,
			block_begin_row, block_begin_col)
	REQUIRE(block_size_row>=1 && block_size_col>=1,
			"Invalid block size (%d, %d)!\n", block_size_row, block_size_col)

	// check if removal of diagonal is required/valid
	if (no_diag && block_size_row!=block_size_col)
	{
		SG_WARNING("Not removing the main diagonal since block is not square!\n");
		no_diag=false;
	}

	SG_DEBUG("Leaving\n");

	return sum<Backend::EIGEN3>(block(kmatrix, block_begin_row, block_begin_col,
				block_size_row, block_size_col), no_diag);
}

SGVector<float64_t> CCustomKernel::row_wise_sum_symmetric_block(index_t
		block_begin, index_t block_size, bool no_diag)
{
	SG_DEBUG("Entering\n");

	if (m_row_subset_stack->has_subsets() || m_col_subset_stack->has_subsets())
	{
		SG_INFO("Row/col subsets initialized! Falling back to "
				"CKernel::row_wise_sum_symmetric_block (slower)!\n");
		return CKernel::row_wise_sum_symmetric_block(block_begin, block_size,
				no_diag);
	}

	REQUIRE(kmatrix.matrix, "The kernel matrix is not initialized!\n")
	REQUIRE(m_is_symmetric, "The kernel matrix is not symmetric!\n")
	REQUIRE(block_begin>=0 && block_begin<kmatrix.num_cols,
			"Invalid block begin index (%d, %d)!\n", block_begin, block_begin)
	REQUIRE(block_begin+block_size<=kmatrix.num_cols,
			"Invalid block size (%d) at starting index (%d, %d)! "
			"Please use smaller blocks!", block_size, block_begin, block_begin)
	REQUIRE(block_size>=1, "Invalid block size (%d)!\n", block_size)

	SGVector<float32_t> s=rowwise_sum<Backend::EIGEN3>(block(kmatrix, block_begin,
				block_begin, block_size, block_size), no_diag);

	// casting to float64_t vector
	SGVector<float64_t> sum(s.vlen);
	for (index_t i=0; i<s.vlen; ++i)
		sum[i]=s[i];

	SG_DEBUG("Leaving\n");

	return sum;
}

SGMatrix<float64_t> CCustomKernel::row_wise_sum_squared_sum_symmetric_block(
		index_t block_begin, index_t block_size, bool no_diag)
{
	SG_DEBUG("Entering\n");

	if (m_row_subset_stack->has_subsets() || m_col_subset_stack->has_subsets())
	{
		SG_INFO("Row/col subsets initialized! Falling back to "
				"CKernel::row_wise_sum_squared_sum_symmetric_block (slower)!\n");
		return CKernel::row_wise_sum_squared_sum_symmetric_block(block_begin,
				block_size, no_diag);
	}

	REQUIRE(kmatrix.matrix, "The kernel matrix is not initialized!\n")
	REQUIRE(m_is_symmetric, "The kernel matrix is not symmetric!\n")
	REQUIRE(block_begin>=0 && block_begin<kmatrix.num_cols,
			"Invalid block begin index (%d, %d)!\n", block_begin, block_begin)
	REQUIRE(block_begin+block_size<=kmatrix.num_cols,
			"Invalid block size (%d) at starting index (%d, %d)! "
			"Please use smaller blocks!", block_size, block_begin, block_begin)
	REQUIRE(block_size>=1, "Invalid block size (%d)!\n", block_size)

	// initialize the matrix that accumulates the row/col-wise sum
	// the first column stores the sum of kernel values
	// the second column stores the sum of squared kernel values
	SGMatrix<float64_t> row_sum(block_size, 2);

	SGVector<float32_t> sum=rowwise_sum<Backend::EIGEN3>(block(kmatrix,
				block_begin, block_begin, block_size, block_size), no_diag);

	SGVector<float32_t> sq_sum=rowwise_sum<Backend::EIGEN3>(
		elementwise_square<Backend::EIGEN3>(block(kmatrix,
		block_begin, block_begin, block_size, block_size)), no_diag);

	for (index_t i=0; i<sum.vlen; ++i)
		row_sum(i, 0)=sum[i];

	for (index_t i=0; i<sq_sum.vlen; ++i)
		row_sum(i, 1)=sq_sum[i];

	SG_DEBUG("Leaving\n");

	return row_sum;
}

SGVector<float64_t> CCustomKernel::row_col_wise_sum_block(index_t
		block_begin_row, index_t block_begin_col, index_t block_size_row,
		index_t block_size_col, bool no_diag)
{
	SG_DEBUG("Entering\n");

	if (m_row_subset_stack->has_subsets() || m_col_subset_stack->has_subsets())
	{
		SG_INFO("Row/col subsets initialized! Falling back to "
				"CKernel::row_col_wise_sum_block (slower)!\n");
		return CKernel::row_col_wise_sum_block(block_begin_row, block_begin_col,
				block_size_row, block_size_col, no_diag);
	}

	REQUIRE(kmatrix.matrix, "The kernel matrix is not initialized!\n")
	REQUIRE(block_begin_row>=0 && block_begin_row<kmatrix.num_rows &&
			block_begin_col>=0 && block_begin_col<kmatrix.num_cols,
			"Invalid block begin index (%d, %d)!\n",
			block_begin_row, block_begin_col)
	REQUIRE(block_begin_row+block_size_row<=kmatrix.num_rows &&
			block_begin_col+block_size_col<=kmatrix.num_cols,
			"Invalid block size (%d, %d) at starting index (%d, %d)! "
			"Please use smaller blocks!", block_size_row, block_size_col,
			block_begin_row, block_begin_col)
	REQUIRE(block_size_row>=1 && block_size_col>=1,
			"Invalid block size (%d, %d)!\n", block_size_row, block_size_col)

	// check if removal of diagonal is required/valid
	if (no_diag && block_size_row!=block_size_col)
	{
		SG_WARNING("Not removing the main diagonal since block is not square!\n");
		no_diag=false;
	}

	// initialize the vector that accumulates the row/col-wise sum
	// the first block_size_row entries store the row-wise sum of kernel values
	// the nextt block_size_col entries store the col-wise sum of kernel values
	SGVector<float64_t> sum(block_size_row+block_size_col);

	SGVector<float32_t> rowwise=rowwise_sum<Backend::EIGEN3>(block(kmatrix,
				block_begin_row, block_begin_col, block_size_row,
				block_size_col), no_diag);

	SGVector<float32_t> colwise=colwise_sum<Backend::EIGEN3>(block(kmatrix,
				block_begin_row, block_begin_col, block_size_row,
				block_size_col), no_diag);

	for (index_t i=0; i<rowwise.vlen; ++i)
		sum[i]=rowwise[i];

	for (index_t i=0; i<colwise.vlen; ++i)
		sum[i+rowwise.vlen]=colwise[i];

	SG_DEBUG("Leaving\n");

	return sum;
}
#endif // HAVE_LINALG_LIB

void CCustomKernel::cleanup_custom()
{
	SG_DEBUG("Entering\n")
	remove_all_row_subsets();
	remove_all_col_subsets();

	kmatrix=SGMatrix<float32_t>();
	upper_diagonal=false;

	SG_DEBUG("Leaving\n")
}

void CCustomKernel::cleanup()
{
	cleanup_custom();
	CKernel::cleanup();
}

void CCustomKernel::add_row_subset(SGVector<index_t> subset)
{
	m_row_subset_stack->add_subset(subset);
	row_subset_changed_post();
}

void CCustomKernel::add_row_subset_in_place(SGVector<index_t> subset)
{
	m_row_subset_stack->add_subset_in_place(subset);
	row_subset_changed_post();
}

void CCustomKernel::remove_row_subset()
{
	m_row_subset_stack->remove_subset();
	row_subset_changed_post();
}

void CCustomKernel::remove_all_row_subsets()
{
	m_row_subset_stack->remove_all_subsets();
	row_subset_changed_post();
}

void CCustomKernel::row_subset_changed_post()
{
	if (m_row_subset_stack->has_subsets())
		num_lhs=m_row_subset_stack->get_size();
	else
		num_lhs=kmatrix.num_rows;
}

void CCustomKernel::add_col_subset(SGVector<index_t> subset)
{
	m_col_subset_stack->add_subset(subset);
	col_subset_changed_post();
}

void CCustomKernel::add_col_subset_in_place(SGVector<index_t> subset)
{
	m_col_subset_stack->add_subset_in_place(subset);
	col_subset_changed_post();
}

void CCustomKernel::remove_col_subset()
{
	m_col_subset_stack->remove_subset();
	col_subset_changed_post();
}

void CCustomKernel::remove_all_col_subsets()
{
	m_col_subset_stack->remove_all_subsets();
	col_subset_changed_post();
}

void CCustomKernel::col_subset_changed_post()
{
	if (m_col_subset_stack->has_subsets())
		num_rhs=m_col_subset_stack->get_size();
	else
		num_rhs=kmatrix.num_cols;
}
