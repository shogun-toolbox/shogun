#include "classifier/svm/MKLRegression.h"
#include "kernel/CombinedKernel.h"

CMKLRegression::CMKLRegression(CSVM* s) : CMKL(s)
{
}

CMKLRegression::~CMKLRegression()
{
}

float64_t CMKLRegression::compute_sum_alpha()
{
	// not correct
	float64_t suma=0;
	int32_t nsv=svm->get_num_support_vectors();
	for (int32_t i=0; i<nsv; i++)
		suma+=CMath::abs(svm->get_alpha(i))*tube_epsilon-svm->get_alpha(i);

	return suma;
}

void CMKLRegression::init_training()
{
	ASSERT(labels);
}

void CMKLRegression::compute_sum_beta(float64_t* sumw)
{
	ASSERT(sumw);

	int32_t nsv=svm->get_num_support_vectors();
	int32_t num_kernels = kernel->get_num_subkernels();
	float64_t* beta = new float64_t[num_kernels];
	int32_t nweights=0;
	const float64_t* old_beta = kernel->get_subkernel_weights(nweights);
	ASSERT(nweights==num_kernels);
	ASSERT(old_beta);

	for (int32_t i=0; i<num_kernels; i++)
	{
		beta[i]=0;
		sumw[i]=0;
	}

	for (int32_t n=0; n<num_kernels; n++)
	{
		beta[n]=1.0;
		kernel->set_subkernel_weights(beta, num_kernels);

		for (int32_t i=0; i<nsv; i++)
		{   
			int32_t ii=svm->get_support_vector(i);

			for (int32_t j=0; j<nsv; j++)
			{   
				int32_t jj=svm->get_support_vector(j);
				sumw[n]+=0.5*svm->get_alpha(i)*svm->get_alpha(j)*kernel->kernel(ii,jj);
			}
		}
		beta[n]=0.0;
	}

	mkl_iterations++;
	kernel->set_subkernel_weights( (float64_t*) old_beta, num_kernels);
}
