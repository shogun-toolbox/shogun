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
	m_parameters->add_matrix(&kmatrix, &num_rows, &num_cols, "kmatrix",
							 "Kernel matrix.");
	m_parameters->add(&upper_diagonal, "upper_diagonal");
}

CCustomKernel::CCustomKernel()
: CKernel(10), kmatrix(NULL), num_rows(0), num_cols(0), upper_diagonal(false)
{
	init();
}

CCustomKernel::CCustomKernel(CKernel* k)
: CKernel(10), kmatrix(NULL), num_rows(0), num_cols(0), upper_diagonal(false)
{
	init();

	if (k->get_lhs_equals_rhs())
	{
		int32_t cols=k->get_num_vec_lhs();
		SG_DEBUG( "using custom kernel of size %dx%d\n", cols,cols);

		kmatrix= new float32_t[(int64_t(cols)+1)*cols/2];

		upper_diagonal=true;
		num_rows=cols;
		num_cols=cols;

		for (int32_t row=0; row<num_rows; row++)
		{
			for (int32_t col=row; col<num_cols; col++)
				kmatrix[int64_t(row) * num_cols - (int64_t(row)+1)*row/2 + col]=k->kernel(row,col);
		}
	}
	else
	{
		int32_t rows=k->get_num_vec_lhs();
		int32_t cols=k->get_num_vec_rhs();
		kmatrix= new float32_t[int64_t(rows)*cols];

		upper_diagonal=false;
		num_rows=rows;
		num_cols=cols;

		for (int32_t row=0; row<num_rows; row++)
		{
			for (int32_t col=0; col<num_cols; col++)
				kmatrix[int64_t(row) * num_cols + col]=k->kernel(row,col);
		}
	}

	dummy_init(num_rows, num_cols);

}

CCustomKernel::CCustomKernel(SGMatrix<float64_t> km)
: CKernel(10), kmatrix(NULL), num_rows(0), num_cols(0), upper_diagonal(false)
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

	SG_DEBUG( "num_vec_lhs: %d vs num_rows %d\n", l->get_num_vectors(), num_rows);
	SG_DEBUG( "num_vec_rhs: %d vs num_cols %d\n", r->get_num_vectors(), num_cols);
	ASSERT(l->get_num_vectors()==num_rows);
	ASSERT(r->get_num_vectors()==num_cols);
	return init_normalizer();
}

void CCustomKernel::cleanup_custom()
{
	SG_DEBUG("cleanup up custom kernel\n");
	delete[] kmatrix;
	kmatrix=NULL;
	upper_diagonal=false;
	num_cols=0;
	num_rows=0;
}

void CCustomKernel::cleanup()
{
	cleanup_custom();
	CKernel::cleanup();
}

