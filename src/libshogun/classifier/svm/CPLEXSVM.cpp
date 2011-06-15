/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 1999-2009 Soeren Sonnenburg
 * Copyright (C) 1999-2009 Fraunhofer Institute FIRST and Max-Planck-Society
 */

#include "classifier/svm/CPLEXSVM.h"
#include "lib/common.h"

#ifdef USE_CPLEX
#include "lib/io.h"
#include "lib/Mathematics.h"
#include "lib/Cplex.h"
#include "features/Labels.h"

using namespace shogun;

CCPLEXSVM::CCPLEXSVM()
: CSVM()
{
}

CCPLEXSVM::~CCPLEXSVM()
{
}

bool CCPLEXSVM::train(CFeatures* data)
{
	bool result = false;
	CCplex cplex;

	if (data)
	{
		if (labels->get_num_labels() != data->get_num_vectors())
			SG_ERROR("Number of training vectors does not match number of labels\n");
		kernel->init(data, data);
	}

	if (cplex.init(E_QP))
	{
		int32_t n,m;
		int32_t num_label=0;
		float64_t* y = labels->get_labels(num_label);
		float64_t* H = kernel->get_kernel_matrix<float64_t>(m, n, NULL);
		ASSERT(n>0 && n==m && n==num_label);
		float64_t* alphas=new float64_t[n];
		float64_t* lb=new float64_t[n];
		float64_t* ub=new float64_t[n];

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
				set_alpha(j, alphas[i]*labels->get_label(i));
				set_support_vector(j, i);
				j++;
			}
		}
		//compute_objective();
		SG_INFO( "obj = %.16f, rho = %.16f\n",get_objective(),get_bias());
		SG_INFO( "Number of SV: %ld\n", get_num_support_vectors());

		delete[] alphas;
		delete[] lb;
		delete[] ub;
		delete[] H;

		result = true;
	}

	if (!result)
		SG_ERROR( "cplex svm failed");

	return result;
}
#endif
