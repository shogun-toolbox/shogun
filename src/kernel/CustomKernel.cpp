/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 2 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 1999-2006 Soeren Sonnenburg
 * Copyright (C) 1999-2006 Fraunhofer Institute FIRST and Max-Planck-Society
 */

#include "lib/common.h"
#include "kernel/CustomKernel.h"
#include "features/Features.h"
#include "lib/io.h"

CCustomKernel::CCustomKernel()
  : CKernel(0),kmatrix(NULL),num_rows(0),num_cols(0),upper_diagonal(false)
{
}

CCustomKernel::~CCustomKernel() 
{
	cleanup();
}

SHORTREAL* CCustomKernel::get_kernel_matrix(int &num_vec1, int &num_vec2, SHORTREAL* target)
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
			CIO::message(M_ERROR, "no features assigned to kernel\n");
			return NULL;
		}
	}
}
  
bool CCustomKernel::init(CFeatures* l, CFeatures* r, bool do_init)
{
	CKernel::init(l, r, do_init); 

	CIO::message(M_DEBUG, "num_vec_lhs: %d vs num_rows %d\n", l->get_num_vectors(), num_rows);
	CIO::message(M_DEBUG, "num_vec_rhs: %d vs num_cols %d\n", r->get_num_vectors(), num_cols);
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

bool CCustomKernel::set_diag_kernel_matrix_from_diag(const DREAL* km, int cols)
{
	cleanup();
	CIO::message(M_DEBUG, "using custom kernel of size %dx%d\n", cols,cols);

	int num=cols*(cols+1)/2;

	kmatrix= new SHORTREAL[num];

	if (kmatrix)
	{
		upper_diagonal=true;
		num_rows=cols;
		num_cols=cols;

		for (INT i=0; i<num; i++)
			kmatrix[i]=km[i];

		return true;
	}
	else
		return false;
}

bool CCustomKernel::set_diag_kernel_matrix_from_full(const DREAL* km, int cols)
{
	cleanup();
	CIO::message(M_DEBUG, "using custom kernel of size %dx%d\n", cols,cols);

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

bool CCustomKernel::set_full_kernel_matrix_from_full(const DREAL* km, int rows, int cols)
{
	cleanup();
	CIO::message(M_DEBUG, "using custom kernel of size %dx%d\n", rows,cols);

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
