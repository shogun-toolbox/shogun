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
#include <shogun/io/SGIO.h>

using namespace shogun;

void
CCustomKernel::init()
{
	m_row_subset=NULL;
	m_col_subset=NULL;
	m_free_km=true;

	SG_ADD((CSGObject**)&m_row_subset, "row_subset", "Subset of rows",
			MS_NOT_AVAILABLE);
	SG_ADD((CSGObject**)&m_col_subset, "col_subset", "Subset of columns",
			MS_NOT_AVAILABLE);
	SG_ADD(&m_free_km, "free_km", "Wheather kernel matrix should be freed in "
			"destructor", MS_NOT_AVAILABLE);

	m_parameters->add(&kmatrix, "kmatrix", "Kernel matrix.");
	m_parameters->add(&upper_diagonal, "upper_diagonal");
}

CCustomKernel::CCustomKernel()
: CKernel(10), kmatrix(), upper_diagonal(false)
{
	init();
}

CCustomKernel::CCustomKernel(CKernel* k)
: CKernel(10)
{
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
	init();
	set_full_kernel_matrix_from_full(km);
}

CCustomKernel::~CCustomKernel()
{
	SG_UNREF(m_row_subset);
	SG_UNREF(m_col_subset);
	cleanup();
}

bool CCustomKernel::dummy_init(int32_t rows, int32_t cols)
{
	return init(new CDummyFeatures(rows), new CDummyFeatures(cols));
}

bool CCustomKernel::init(CFeatures* l, CFeatures* r)
{
	remove_row_subset();
	remove_col_subset();

	/* make it possible to call with NULL values since features are useless
	 * for custom kernel matrix */
	if (!l)
		l=lhs;

	if (!r)
		r=rhs;

	CKernel::init(l, r);

	SG_DEBUG( "num_vec_lhs: %d vs num_rows %d\n", l->get_num_vectors(), kmatrix.num_rows);
	SG_DEBUG( "num_vec_rhs: %d vs num_cols %d\n", r->get_num_vectors(), kmatrix.num_cols);
	ASSERT(l->get_num_vectors()==kmatrix.num_rows);
	ASSERT(r->get_num_vectors()==kmatrix.num_cols);
	return init_normalizer();
}

void CCustomKernel::cleanup_custom()
{
	SG_DEBUG("cleanup up custom kernel\n");
	remove_row_subset();
	remove_col_subset();

	if (m_free_km)
		SG_FREE(kmatrix.matrix);

	kmatrix.matrix=NULL;
	upper_diagonal=false;
	kmatrix.num_cols=0;
	kmatrix.num_rows=0;
}

void CCustomKernel::cleanup()
{
	remove_row_subset();
	remove_col_subset();
	cleanup_custom();
	CKernel::cleanup();
}

void CCustomKernel::set_row_subset(CSubset* subset)
{
	SG_UNREF(m_row_subset);
	m_row_subset=subset;
	SG_REF(subset);

	/* update num_lhs */
	num_lhs=subset ? subset->get_size() : 0;
}
void CCustomKernel::set_col_subset(CSubset* subset)
{
	SG_UNREF(m_col_subset);
	m_col_subset=subset;
	SG_REF(subset);

	/* update num_rhs */
	num_rhs=subset ? subset->get_size() : 0;
}

void CCustomKernel::remove_row_subset()
{
	set_row_subset(NULL);

	/* restore num_lhs */
	num_lhs=kmatrix.num_rows;
}

void CCustomKernel::remove_col_subset()
{
	set_col_subset(NULL);

	/* restore num_rhs */
	num_rhs=kmatrix.num_cols;
}

void CCustomKernel::print_kernel_matrix(const char* prefix) const
{
	index_t num_rows=m_row_subset ? m_row_subset->get_size() : kmatrix.num_rows;
	index_t num_cols=m_col_subset ? m_col_subset->get_size() : kmatrix.num_cols;
	for (index_t i=0; i<num_rows; ++i)
	{
		for (index_t j=0; j<num_cols; ++j)
		{
			index_t real_i=row_subset_idx_conversion(i);
			index_t real_j=col_subset_idx_conversion(j);
			SG_PRINT("%s%4.2f", kmatrix.matrix[kmatrix.num_rows*real_j+real_i],
					prefix);
			if (j<num_cols-1)
				SG_PRINT(", \t");
		}
		SG_PRINT("\n");
	}
}
