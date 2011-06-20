/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 1999-2009 Soeren Sonnenburg
 * Copyright (C) 1999-2009 Fraunhofer Institute FIRST and Max-Planck-Society
 */

#include "lib/common.h"
#include "kernel/CustomKernel.h"
#include "features/Features.h"
#include "features/DummyFeatures.h"
#include "lib/io.h"

using namespace shogun;

void
CCustomKernel::init(void)
{
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
	cleanup();
}

bool CCustomKernel::dummy_init(int32_t rows, int32_t cols)
{
	return init(new CDummyFeatures(rows), new CDummyFeatures(cols));
}

bool CCustomKernel::init(CFeatures* l, CFeatures* r)
{
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
	delete[] kmatrix.matrix;
	kmatrix.matrix=NULL;
	upper_diagonal=false;
	kmatrix.num_cols=0;
	kmatrix.num_rows=0;
}

void CCustomKernel::cleanup()
{
	cleanup_custom();
	CKernel::cleanup();
}

