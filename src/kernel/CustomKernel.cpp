#include "lib/common.h"
#include "kernel/CustomKernel.h"
#include "features/Features.h"
#include "lib/io.h"

#include <assert.h>

CCustomKernel::CCustomKernel()
  : CKernel(0),kmatrix(NULL),num_rows(0),num_cols(0)
{
}

CCustomKernel::~CCustomKernel() 
{
	cleanup();
}
  
bool CCustomKernel::init(CFeatures* l, CFeatures* r, bool do_init)
{
	CKernel::init(l, r, do_init); 

	CIO::message(M_DEBUG, "num_vec_lhs: %d vs num_rows %d\n", l->get_num_vectors(), num_rows);
	CIO::message(M_DEBUG, "num_vec_rhs: %d vs num_cols %d\n", r->get_num_vectors(), num_cols);
	assert(l->get_num_vectors() == num_rows);
	assert(r->get_num_vectors() == num_cols);
	return true;
}

void CCustomKernel::cleanup()
{
	delete[] kmatrix;
	kmatrix=NULL;
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

bool CCustomKernel::set_kernel_matrix_diag(const REAL* km, int rows, int cols)
{
	cleanup();
	CIO::message(M_DEBUG, "using custom kernel of size %dx%d\n", rows,cols);

	int l=CMath::min(rows,cols);
	int u=CMath::max(rows,cols);
	int num=l*(l+1)/2 + (u-l)*l;

	kmatrix= new SHORTREAL[num];

	if (kmatrix)
	{
		num_rows=rows;
		num_cols=cols;

		for (INT i=0; i<num; i++)
			kmatrix[i]=km[i];

		return true;
	}
	else
		return false;
}

bool CCustomKernel::set_kernel_matrix(const REAL* km, int rows, int cols)
{
	cleanup();
	CIO::message(M_DEBUG, "using custom kernel of size %dx%d\n", rows,cols);

	num_rows=rows;
	num_cols=cols;
	kmatrix= new SHORTREAL[rows*cols];

	if (kmatrix)
	{
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
	/*
	int l=CMath::min(rows,cols);
	int u=CMath::max(rows,cols);

	kmatrix= new SHORTREAL[l*(l+1)/2 + (u-l)*l];

	if (kmatrix)
	{
		num_rows=rows;
		num_cols=cols;

		if (num_rows < num_cols)
		{
			for (INT row=0; row<num_rows; row++)
			{
				for (INT col=row; col<num_cols; col++)
				{
					kmatrix[row * num_cols - row*(row+1)/2 + col]=km[row*num_cols+col];
				}
			}
		}
		else
		{
			for (INT row=0; row<num_rows; row++)
			{
				for (INT col=0; col<row && col<num_cols; col++)
				{
					INT r = CMath::min(row, num_cols-1);
					kmatrix[row * num_cols - r*(r+1)/2 + col]=km[row*num_cols+col];
				}
			}
		}
		return true;
	}
	else
		return false;
		*/
}
