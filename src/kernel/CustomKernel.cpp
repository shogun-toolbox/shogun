#include "lib/common.h"
#include "kernel/CustomKernel.h"
#include "features/Features.h"
#include "lib/io.h"

#include <assert.h>

CCustomKernel::CCustomKernel()
  : CKernel(0),kmatrix(NULL),num_cols(0)
{
}

CCustomKernel::~CCustomKernel() 
{
	cleanup();
}
  
bool CCustomKernel::init(CFeatures* l, CFeatures* r, bool do_init)
{
	CKernel::init(l, r, do_init); 
	return true;
}

void CCustomKernel::cleanup()
{
	delete[] kmatrix;
	kmatrix=NULL;
	num_cols=0;
}

bool CCustomKernel::load_init(FILE* src)
{
	return false;
}

bool CCustomKernel::save_init(FILE* dest)
{
	return false;
}

bool CCustomKernel::set_kernel_matrix_diag(const REAL* m, int num)
{
	kmatrix= new SHORTREAL[num*num/2];

	if (kmatrix)
	{
		num_cols=num;
		for (INT i=0; i<num*num/2; i++)
			kmatrix[i]=m[i];

		return true;
	}
	else
		return false;
}

bool CCustomKernel::set_kernel_matrix(const REAL* m, int num)
{
	kmatrix= new SHORTREAL[num*num/2];

	if (kmatrix)
	{
		num_cols=num;
		for (INT i=0; i<num; i++)
			for (INT j=i; j<num; j++)
				kmatrix[i*(i+1)/2 + j]=m[i*num+j];

		return true;
	}
	else
		return false;
}
