/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Copyright (W) 2012 Sergey Lisitsyn
 */

#include <statistics/KernelMeanMatching.h>
#include <lib/external/libqp.h>


static float64_t* kmm_K = NULL;
static int32_t kmm_K_ld = 0;

static const float64_t* kmm_get_col(uint32_t i)
{
	return kmm_K + kmm_K_ld*i;
}

namespace shogun
{
CKernelMeanMatching::CKernelMeanMatching() :
	CSGObject(), m_kernel(NULL)
{
}

CKernelMeanMatching::CKernelMeanMatching(CKernel* kernel, SGVector<index_t> training_indices,
                                         SGVector<index_t> test_indices) :
	CSGObject(), m_kernel(NULL)
{
	set_kernel(kernel);
	set_training_indices(training_indices);
	set_test_indices(test_indices);
}

SGVector<float64_t> CKernelMeanMatching::compute_weights()
{
	int32_t i,j;
	ASSERT(m_kernel)
	ASSERT(m_training_indices.vlen)
	ASSERT(m_test_indices.vlen)

	int32_t n_tr = m_training_indices.vlen;
	int32_t n_te = m_test_indices.vlen;

	SGVector<float64_t> weights(n_tr);
	weights.zero();

	kmm_K = SG_MALLOC(float64_t, n_tr*n_tr);
	kmm_K_ld = n_tr;
	float64_t* diag_K = SG_MALLOC(float64_t, n_tr);
	for (i=0; i<n_tr; i++)
	{
		float64_t d = m_kernel->kernel(m_training_indices[i], m_training_indices[i]);
		diag_K[i] = d;
		kmm_K[i*n_tr+i] = d;
		for (j=i+1; j<n_tr; j++)
		{
			d = m_kernel->kernel(m_training_indices[i],m_training_indices[j]);
			kmm_K[i*n_tr+j] = d;
			kmm_K[j*n_tr+i] = d;
		}
	}
	float64_t* kappa = SG_MALLOC(float64_t, n_tr);
	for (i=0; i<n_tr; i++)
	{
		float64_t avg = 0.0;
		for (j=0; j<n_te; j++)
			avg+= m_kernel->kernel(m_training_indices[i],m_test_indices[j]);

		avg *= float64_t(n_tr)/n_te;
		kappa[i] = -avg;
	}
	float64_t* a = SG_MALLOC(float64_t, n_tr);
	for (i=0; i<n_tr; i++) a[i] = 1.0;
	float64_t* LB = SG_MALLOC(float64_t, n_tr);
	float64_t* UB = SG_MALLOC(float64_t, n_tr);
	float64_t B = 2.0;
	for (i=0; i<n_tr; i++)
	{
		LB[i] = 0.0;
		UB[i] = B;
	}
	for (i=0; i<n_tr; i++)
		weights[i] = 1.0/float64_t(n_tr);

	libqp_state_T result =
		libqp_gsmo_solver(&kmm_get_col,diag_K,kappa,a,1.0,LB,UB,weights,n_tr,1000,1e-9,NULL);

	SG_DEBUG("libqp exitflag=%d, %d iterations passed, primal objective=%f\n",
	         result.exitflag,result.nIter,result.QP);

	SG_FREE(kappa);
	SG_FREE(a);
	SG_FREE(LB);
	SG_FREE(UB);
	SG_FREE(diag_K);
	SG_FREE(kmm_K);

	return weights;
}

}
