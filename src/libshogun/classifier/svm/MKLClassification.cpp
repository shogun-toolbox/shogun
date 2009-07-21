#include "classifier/svm/MKLClassification.h"
#include "kernel/CombinedKernel.h"

CMKLClassification::CMKLClassification(CSVM* s) : CMKL(s)
{
}

CMKLClassification::~CMKLClassification()
{
}
float64_t CMKLClassification::compute_sum_alpha()
{
	float64_t suma=0;
	int32_t nsv=svm->get_num_support_vectors();
	for (int32_t i=0; i<nsv; i++)
		suma+=CMath::abs(svm->get_alpha(i));

	return suma;
}

void CMKLClassification::init_training()
{
	ASSERT(labels && labels->is_two_class_labeling());
}

void CMKLClassification::compute_sum_beta(float64_t* sumw)
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


struct S_THREAD_PARAM 
{
	float64_t * lin ;
	float64_t* W;
	int32_t start, end;
	int32_t * active2dnum ;
	int32_t * docs ;
	CKernel* kernel ;
};

// assumes that all constraints are satisfied
float64_t CMKLClassification::compute_mkl_dual_objective()
{
	int32_t n=get_num_support_vectors();
	float64_t mkl_obj=0;

	if (labels && kernel && kernel->get_kernel_type() == K_COMBINED)
	{
		CKernel* kn = ((CCombinedKernel*)kernel)->get_first_kernel();
		while (kn)
		{
			float64_t sum=0;
			for (int32_t i=0; i<n; i++)
			{
				int32_t ii=get_support_vector(i);

				for (int32_t j=0; j<n; j++)
				{
					int32_t jj=get_support_vector(j);
					sum+=get_alpha(i)*get_alpha(j)*kn->kernel(ii,jj);
				}
			}

			if (mkl_norm==1.0)
				mkl_obj = CMath::max(mkl_obj, sum);
			else
				mkl_obj += CMath::pow(sum, mkl_norm/(mkl_norm-1));

			kn = ((CCombinedKernel*) kernel)->get_next_kernel();
		}

		if (mkl_norm==1.0)
			mkl_obj=-0.5*mkl_obj;
		else
			mkl_obj= -0.5*CMath::pow(mkl_obj, (mkl_norm-1)/mkl_norm);

		for (int32_t i=0; i<n; i++)
		{
			int32_t ii=get_support_vector(i);
			mkl_obj+=get_alpha(i)*labels->get_label(ii);
		}
	}
	else
		SG_ERROR( "cannot compute objective, labels or kernel not set\n");

	return -mkl_obj;
}
