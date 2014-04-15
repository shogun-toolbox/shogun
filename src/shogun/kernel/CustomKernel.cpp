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
#include <shogun/base/ParameterMap.h>

using namespace shogun;

void
CCustomKernel::init()
{
	m_row_subset_stack=new CSubsetStack();
	SG_REF(m_row_subset_stack)
	m_col_subset_stack=new CSubsetStack();
	SG_REF(m_col_subset_stack)
	m_free_km=true;

	SG_ADD((CSGObject**)&m_row_subset_stack, "row_subset_stack",
			"Subset stack of rows", MS_NOT_AVAILABLE);
	SG_ADD((CSGObject**)&m_col_subset_stack, "col_subset_stack",
			"Subset stack of columns", MS_NOT_AVAILABLE);
	SG_ADD(&m_free_km, "free_km", "Whether kernel matrix should be freed in "
			"destructor", MS_NOT_AVAILABLE);
	SG_ADD(&kmatrix, "kmatrix", "Kernel matrix.", MS_NOT_AVAILABLE);
	SG_ADD(&upper_diagonal, "upper_diagonal", "Upper diagonal", MS_NOT_AVAILABLE);

	/* new parameter from param version 0 to 1 */
	m_parameter_map->put(
			new SGParamInfo("free_km", CT_SCALAR, ST_NONE, PT_BOOL, 1),
			new SGParamInfo()
	);

	/* new parameter from param version 0 to 1 */
	m_parameter_map->put(
			new SGParamInfo("row_subset_stack", CT_SCALAR, ST_NONE, PT_SGOBJECT, 1),
			new SGParamInfo()
	);

	/* new parameter from param version 0 to 1 */
	m_parameter_map->put(
			new SGParamInfo("col_subset_stack", CT_SCALAR, ST_NONE, PT_SGOBJECT, 1),
			new SGParamInfo()
	);
	m_parameter_map->finalize_map();
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
		set_full_kernel_matrix_from_full(casted->get_float32_kernel_matrix());
		m_free_km=false;
	}
	else
		set_full_kernel_matrix_from_full(k->get_kernel_matrix());
}

CCustomKernel::CCustomKernel(SGMatrix<float64_t> km)
: CKernel(10), upper_diagonal(false)
{
	SG_DEBUG("Entering CCustomKernel::CCustomKernel(SGMatrix<float64_t>)\n")
	init();
	set_full_kernel_matrix_from_full(km);
	SG_DEBUG("Leaving CCustomKernel::CCustomKernel(SGMatrix<float64_t>)\n")
}

CCustomKernel::CCustomKernel(SGMatrix<float32_t> km)
: CKernel(10), upper_diagonal(false)
{
	SG_DEBUG("Entering CCustomKernel::CCustomKernel(SGMatrix<float64_t>)\n")
	init();
	set_full_kernel_matrix_from_full(km);
	SG_DEBUG("Leaving CCustomKernel::CCustomKernel(SGMatrix<float64_t>)\n")
}

CCustomKernel::~CCustomKernel()
{
	SG_DEBUG("Entering CCustomKernel::~CCustomKernel()\n")
	cleanup();
	SG_UNREF(m_row_subset_stack);
	SG_UNREF(m_col_subset_stack);
	SG_DEBUG("Leaving CCustomKernel::~CCustomKernel()\n")
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

		return true;
	}

	/* For other types of CFeatures do the default actions below */
	CKernel::init(l, r);

	SG_DEBUG("num_vec_lhs: %d vs num_rows %d\n", l->get_num_vectors(), kmatrix.num_rows)
	SG_DEBUG("num_vec_rhs: %d vs num_cols %d\n", r->get_num_vectors(), kmatrix.num_cols)
	ASSERT(l->get_num_vectors()==kmatrix.num_rows)
	ASSERT(r->get_num_vectors()==kmatrix.num_cols)
	return init_normalizer();
}

void CCustomKernel::cleanup_custom()
{
	SG_DEBUG("Entering CCustomKernel::cleanup_custom()\n")
	remove_all_row_subsets();
	remove_all_col_subsets();

	kmatrix=SGMatrix<float32_t>();
	upper_diagonal=false;

	SG_DEBUG("Leaving CCustomKernel::cleanup_custom()\n")
}

void CCustomKernel::cleanup()
{
	remove_all_row_subsets();
	remove_all_col_subsets();
	cleanup_custom();
	CKernel::cleanup();
}

void CCustomKernel::add_row_subset(SGVector<index_t> subset)
{
	m_row_subset_stack->add_subset(subset);
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
