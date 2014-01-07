/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 1999-2009 Soeren Sonnenburg
 * Copyright (C) 1999-2009 Fraunhofer Institute FIRST and Max-Planck-Society
 */

#include <classifier/svm/CPLEXSVM.h>
#include <lib/common.h>

#ifdef USE_CPLEX
#include <io/SGIO.h>
#include <mathematics/Math.h>
#include <mathematics/Cplex.h>
#include <labels/Labels.h>

using namespace shogun;

CCPLEXSVM::CCPLEXSVM()
: CSVM()
{
}

CCPLEXSVM::~CCPLEXSVM()
{
}

bool CCPLEXSVM::train_machine(CFeatures* data)
{
	ASSERT(m_labels)
	ASSERT(m_labels->get_label_type() == LT_BINARY)

	bool result = false;
	CCplex cplex;

	if (data)
	{
		if (m_labels->get_num_labels() != data->get_num_vectors())
		{
			SG_ERROR("%s::train_machine(): Number of training vectors (%d) does"
					" not match number of labels (%d)\n", get_name(),
					data->get_num_vectors(), m_labels->get_num_labels());
		}
		kernel->init(data, data);
	}

	if (cplex.init(E_QP))
	{
		int32_t n,m;
		int32_t num_label=0;
		SGVector<float64_t> y=((CBinaryLabels*)m_labels)->get_labels();
		SGMatrix<float64_t> H=kernel->get_kernel_matrix();
		m=H.num_rows;
		n=H.num_cols;
		ASSERT(n>0 && n==m && n==num_label)
		float64_t* alphas=SG_MALLOC(float64_t, n);
		float64_t* lb=SG_MALLOC(float64_t, n);
		float64_t* ub=SG_MALLOC(float64_t, n);

		//hessian y'y.*K
		for (int32_t i=0; i<n; i++)
		{
			lb[i]=0;
			ub[i]=get_C1();

			for (int32_t j=0; j<n; j++)
				H[i*n+j]*=y[j]*y[i];
		}

		//feed qp to cplex


		int32_t j=0;
		for (int32_t i=0; i<n; i++)
		{
			if (alphas[i]>0)
			{
				//set_alpha(j, alphas[i]*labels->get_label(i)/etas[1]);
				set_alpha(j, alphas[i]*((CBinaryLabels*) m_labels)->get_int_label(i));
				set_support_vector(j, i);
				j++;
			}
		}
		//compute_objective();
		SG_INFO("obj = %.16f, rho = %.16f\n",get_objective(),get_bias())
		SG_INFO("Number of SV: %ld\n", get_num_support_vectors())

		SG_FREE(alphas);
		SG_FREE(lb);
		SG_FREE(ub);

		result = true;
	}

	if (!result)
		SG_ERROR("cplex svm failed")

	return result;
}
#endif
