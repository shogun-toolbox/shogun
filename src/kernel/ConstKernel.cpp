#include "lib/common.h"
#include "kernel/ConstKernel.h"
#include "features/Features.h"
#include "lib/io.h"

CConstKernel::CConstKernel(DREAL c) : CKernel(0), const_value(c)
{
}

CConstKernel::~CConstKernel() 
{
}

bool CConstKernel::init(CFeatures* l, CFeatures* r, bool do_init)
{
	CKernel::init(l, r, do_init); 
	return true;
}

bool CConstKernel::load_init(FILE* src)
{
	return false;
}

bool CConstKernel::save_init(FILE* dest)
{
	return false;
}
