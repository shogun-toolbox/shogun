#include "lib/config.h"

#ifdef HAVE_LAPACK
extern "C" {
#include <cblas.h>
}
#endif

#include "lib/common.h"
#include "kernel/DiagKernel.h"
#include "lib/io.h"

#include <assert.h>

CDiagKernel::CDiagKernel(LONG size, REAL d)
  : CKernel(size),diag(d)
{
}

CDiagKernel::~CDiagKernel() 
{
}
  
void CDiagKernel::cleanup()
{
}

bool CDiagKernel::load_init(FILE* src)
{
	return false;
}

bool CDiagKernel::save_init(FILE* dest)
{
	return false;
}

