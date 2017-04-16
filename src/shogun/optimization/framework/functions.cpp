#include <shogun/optimization/framework/functions.h>
using namespace shogun;

func::func()
{
	H = NULL;
	f = NULL;
	x = NULL;
	diag_H = NULL;
	A = NULL;
	dim = 0;
	subgrad = NULL;
}

int func::init(uint32_t dims, uint32_t bsize)
{
	/* Flags : -1 - Not Successful (Memory error)
	 * 			0 - Successful
	 */
	ASSERT(dims > 0);
	ASSERT(bsize > 0);
	REQUIRE(bsize < (std::numeric_limits<size_t>::max() / dims),
		"overflow: %u * %u > %u -- biggest possible BufSize=%u or nDim=%u\n",
		bsize, dims, std::numeric_limits<size_t>::max(),
		(std::numeric_limits<size_t>::max() / dims),
		(std::numeric_limits<size_t>::max() / bsize));
	
	BufSize = bsize;
	dim = dims;
	H = (float64_t*) BMRM_CALLOC( BufSize*BufSize, float64_t);
	f = (float64_t*) BMRM_CALLOC( BufSize, float64_t);
	diag_H = (float64_t*) BMRM_CALLOC( BufSize, float64_t);
	x = (float64_t*) BMRM_CALLOC( BufSize, float64_t);
	A = (float64_t*) BMRM_CALLOC( BufSize*dims, float64_t);
	subgrad = (float64_t*) BMRM_CALLOC( dims, float64_t);
	
	if(H == NULL || diag_H == NULL || f == NULL || x == NULL || subgrad == NULL || A == NULL)
		return -1;
	else
		return 0;
}

void func::cleanup()
{
	BMRM_FREE(H);
	BMRM_FREE(f);
	BMRM_FREE(diag_H);
	BMRM_FREE(x);
	BMRM_FREE(A);
	BMRM_FREE(subgrad);
}
