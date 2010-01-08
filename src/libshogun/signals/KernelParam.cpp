#include <stdio.h>
#include <string.h>

#include "lib/Mathematics.h"
#include "lib/config.h"
#include "lib/io.h"
#include "signals/KernelParam.h"

#include <ui/GUIKernel.h>
#include <shogun/kernel/WeightedDegreePositionStringKernel.h>
#include <shogun/kernel/WeightedDegreeStringKernel.h>
#include <shogun/kernel/CommWordStringKernel.h>
#include <shogun/kernel/WeightedCommWordStringKernel.h>
#include <shogun/kernel/LinearKernel.h>
#include <shogun/kernel/SparseLinearKernel.h>
#include <shogun/kernel/CombinedKernel.h>
#include <shogun/kernel/CustomKernel.h>
#include <shogun/kernel/SalzbergWordStringKernel.h>
#include <shogun/features/SimpleFeatures.h>
//#include <shogun/features/PolyFeatures.h>
#include <shogun/preproc/SortWordString.h>

#define MAXLINE 1000

CKernelParam::CKernelParam()
:CSGObject(), 
	size(1),
	order(3),
	max_mismatch(0),
	use_normalization(true),
	mkl_stepsize(1),
	block_computation(true),
	single_degree(-1),
	length(0),
	center(0),
	step(1),
	kernelname(NULL)

{
}
CKernelParam::~CKernelParam()
{
}

CKernel* CKernelParam::create_kernel(CGUIKernel* ui_kernel)
{
	CKernel* kernel = NULL;

	if (strcmp(kernelname, "WEIGHTEDDEGREEPOS")==0)
        {
		int32_t i=0;
		int32_t* shifts=new int32_t[length];
	
		for (i=center; i<length; i++)
			shifts[i]=(int32_t) floor(((float64_t) (i-center))/step);
	
		for (i=center-1; i>=0; i--)
			shifts[i]=(int32_t) floor(((float64_t) (center-i))/step);
	
		for (i=0; i<length; i++)
		{
			if (shifts[i]>length)
				shifts[i]=length;
		}
	
		for (i=0; i<length; i++)
			SG_INFO( "shift[%i]=%i\n", i, shifts[i]);
	
		float64_t* weights=get_weights();
	
		CKernel* kern=new CWeightedDegreePositionStringKernel(size, weights, order, max_mismatch, shifts, length);
		if (!kern)
			SG_ERROR("Couldn't create WeightedDegreePositionStringKernel with size %d, order %d, max_mismatch %d, length %d, center %d, step %f.\n", size, order, max_mismatch, length, center, step);
		else
			SG_DEBUG("created WeightedDegreePositionStringKernel with size %d, order %d, max_mismatch %d, length %d, center %d, step %f.\n", kern, size, order, max_mismatch, length, center, step);

		delete[] weights;
		delete[] shifts;
		return kern;
        }
	/*else if (strcmp(kernelname, "WEIGHTEDDEGREE")==0)
        {
               kernel=ui_kernel->create_weighteddegreestring(
                        size, order, max_mismatch, use_normalization,
                        mkl_stepsize, block_computation, single_degree);
        }
	*/
	return kernel;

}
float64_t* CKernelParam::get_weights()
{
	float64_t *weights=new float64_t[order*(1+max_mismatch)];
	float64_t sum=0;
	int32_t i=0;

	for (i=0; i<order; i++)
	{
		weights[i]=order-i;
		sum+=weights[i];
	}
	for (i=0; i<order; i++)
		weights[i]/=sum;
	
	for (i=0; i<order; i++)
	{
		for (int32_t j=1; j<=max_mismatch; j++)
		{
			if (j<i+1)
			{
				int32_t nk=CMath::nchoosek(i+1, j);
				weights[i+j*order]=weights[i]/(nk*CMath::pow(3, j));
			}
			else
				weights[i+j*order]=0;
		}
	}

	return weights;
}
