#include "lib/common.h"
#include "lib/io.h"
#include "kernel/Kernel.h"
#include "kernel/OptimizableKernel.h"

COptimizableKernel::COptimizableKernel()
:initialized(false) 
{ 
} ;

COptimizableKernel::~COptimizableKernel() 
{ 
	if (get_is_initialized()) 
		CIO::message(M_ERROR, "COptimizableKernel still initialized on destruction") ;
} ;

bool COptimizableKernel::is_optimizable(CKernel *k)
{
	if ((k!=NULL) && ((k->get_kernel_type() & K_OPTIMIZABLE)!=0))
		return true ;
	return false ;
} ;

bool COptimizableKernel::init_optimization(INT count, INT *IDX, REAL * weights)
{
	CIO::message(M_ERROR, "kernel optimization not implemented\n") ;
	return false ;
} ;

void COptimizableKernel::delete_optimization() 
{
	CIO::message(M_ERROR, "kernel optimization not implemented\n") ;
} ;

REAL COptimizableKernel::compute_optimized(INT idx)
{
	CIO::message(M_ERROR, "kernel optimization not implemented\n") ;
	return 0 ;
} ;
