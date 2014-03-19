#include <shogun/optimization/constraints.h>
using namespace shogun;

constraints::constraints()
{
	TolRel = 0.0;
	TolAbs = 0.0;
	S = NULL;
	I = NULL;
	UB = NULL;
	dim = 0;
	cp_models = 1;
	BufSize = 0;
}

int constraints::init(float64_t trel, 
					  float64_t tabs, 
					  uint32_t dims,
					  uint32_t bsize,
					  uint32_t cpm)
{
	TolRel = trel;
	TolAbs = tabs;
	cp_models = cpm;
	BufSize = bsize;
	dim = dims;
	S = (uint8_t*) BMRM_CALLOC( cpm, uint8_t);
	UB = (float64_t*) BMRM_CALLOC( BufSize, float64_t);
	I = (uint32_t*) BMRM_CALLOC( BufSize, uint32_t);
	if(S == NULL || UB == NULL || I == NULL)
	{
		return -1;
	}
	else
		return 0;
}

void constraints::cleanup()
{
	BMRM_FREE(S);
	BMRM_FREE(I);
	BMRM_FREE(UB);
}
