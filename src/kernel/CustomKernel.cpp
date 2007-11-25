/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 2 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 1999-2007 Soeren Sonnenburg
 * Copyright (C) 1999-2007 Fraunhofer Institute FIRST and Max-Planck-Society
 */

#include "lib/common.h"
#include "kernel/CustomKernel.h"
#include "features/Features.h"
#include "lib/io.h"

CCustomKernel::CCustomKernel()
	: CKernel(10),kmatrix(NULL),num_rows(0),num_cols(0),upper_diagonal(false)
{
}

CCustomKernel::CCustomKernel(CFeatures* l, CFeatures* r)
	: CKernel(10),kmatrix(NULL),num_rows(0),num_cols(0),upper_diagonal(false)
{
	num_rows=l->get_num_vectors();
	num_cols=r->get_num_vectors();
	init(l, r);
}

CCustomKernel::~CCustomKernel()
{
	cleanup();
}

SHORTREAL* CCustomKernel::get_kernel_matrix_shortreal(INT &num_vec1, INT &num_vec2, SHORTREAL* target)
{
	if (target == NULL)
		return CKernel::get_kernel_matrix_shortreal(num_vec1, num_vec2, target);
	else
	{
		CFeatures* f1 = get_lhs();
		CFeatures* f2 = get_rhs();
		if (f1 && f2)
		{
			num_vec1=f1->get_num_vectors();
			num_vec2=f2->get_num_vectors();
			return kmatrix;
		}
		else
		{
         SG_ERROR( "no features assigned to kernel\n");
			return NULL;
		}
	}
}
  
bool CCustomKernel::init(CFeatures* l, CFeatures* r)
{
	CKernel::init(l, r);

	SG_DEBUG( "num_vec_lhs: %d vs num_rows %d\n", l->get_num_vectors(), num_rows);
	SG_DEBUG( "num_vec_rhs: %d vs num_cols %d\n", r->get_num_vectors(), num_cols);
	ASSERT(l->get_num_vectors() == num_rows);
	ASSERT(r->get_num_vectors() == num_cols);
	return true;
}

void CCustomKernel::cleanup()
{
	delete[] kmatrix;
	kmatrix=NULL;
	upper_diagonal=false;
	num_cols=0;
	num_rows=0;
}

bool CCustomKernel::load_init(FILE* src)
{
	return false;
}

bool CCustomKernel::save_init(FILE* dest)
{

	return false;
}

bool CCustomKernel::set_triangle_kernel_matrix_from_triangle(const DREAL* km, int len)
{
	ASSERT(km);
	ASSERT(len > 0);

	INT cols = floor(-0.5 + CMath::sqrt(0.25+2*len));
	if (cols*(cols+1)/2 != len)
	{
		SG_ERROR("km should be a vector containing a lower triangle matrix, with len=cols*(cols+1)/2 elements\n");
		return false;
	}


	cleanup();
	SG_DEBUG( "using custom kernel of size %dx%d\n", cols,cols);

	kmatrix= new SHORTREAL[len];

	if (kmatrix)
	{
		upper_diagonal=true;
		num_rows=cols;
		num_cols=cols;

		for (INT i=0; i<len; i++)
			kmatrix[i]=km[i];

		return true;
	}
	else
		return false;
}

bool CCustomKernel::set_triangle_kernel_matrix_from_full(const DREAL* km, INT rows, INT cols)
{
	ASSERT(rows == cols);

	cleanup();
	SG_DEBUG( "using custom kernel of size %dx%d\n", cols,cols);

	kmatrix= new SHORTREAL[cols*(cols+1)/2];

	if (kmatrix)
	{
		upper_diagonal=true;
		num_rows=cols;
		num_cols=cols;

		for (INT row=0; row<num_rows; row++)
		{
			for (INT col=row; col<num_cols; col++)
			{
				kmatrix[row * num_cols - row*(row+1)/2 + col]=km[col*num_rows+row];
			}
		}
		return true;
	}
	else
		return false;
}

bool CCustomKernel::set_full_kernel_matrix_from_full(const DREAL* km, INT rows, INT cols)
{
	cleanup();
	SG_DEBUG( "using custom kernel of size %dx%d\n", rows,cols);

	kmatrix= new SHORTREAL[rows*cols];

	if (kmatrix)
	{
		upper_diagonal=false;
		num_rows=rows;
		num_cols=cols;

		for (INT row=0; row<num_rows; row++)
		{
			for (INT col=0; col<num_cols; col++)
			{
				kmatrix[row * num_cols + col]=km[col*num_rows+row];
			}
		}
		return true;
	}
	else
		return false;
}
