/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 2 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 1999-2007 Soeren Sonnenburg
 * Copyright (C) 1999-2007 Fraunhofer Institute FIRST and Max-Planck-Society
 */

#include "classifier/svm/CPLEXSVM.h"
#include "lib/common.h"
#ifdef USE_CPLEX
#include "lib/io.h"
#include "lib/Mathematics.h"
#include "lib/Cplex.h"
#include "features/Labels.h"

CCPLEXSVM::CCPLEXSVM()
{
}

CCPLEXSVM::~CCPLEXSVM()
{
}

bool CCPLEXSVM::train()
{
	bool result = false;
	CLabels* lab = CKernelMachine::get_labels();
	CCplex cplex;

	if (cplex.init(E_QP))
	{
		INT n,m;
		INT num_label=0;
		DREAL* y = lab->get_labels(num_label);
		DREAL* H = kernel->get_kernel_matrix_real(m, n, NULL);
		ASSERT(n>0 && n==m && n==num_label);
		DREAL* alphas=new DREAL[n];
		DREAL* lb=new DREAL[n];
		DREAL* ub=new DREAL[n];
		ASSERT(lb && ub);

		//hessian y'y.*K
		for (int i=0; i<n; i++)
		{
			lb[i]=0;
			ub[i]=get_C1();

			for (int j=0; j<n; j++)
				H[i*n+j]*=y[j]*y[i];
		}

		//feed qp to cplex


		int j=0;
		for (int i=0; i<n; i++)
		{
			if (alphas[i]>0)
			{
				//set_alpha(j, alphas[i]*lab->get_label(i)/etas[1]);
				set_alpha(j, alphas[i]*lab->get_label(i));
				set_support_vector(j, i);
				j++;
			}
		}
		compute_objective();
		SG_INFO( "obj = %.16f, rho = %.16f\n",get_objective(),get_bias());
		SG_INFO( "Number of SV: %ld\n", get_num_support_vectors());

		delete[] alphas;
		delete[] lb;
		delete[] ub;
		delete[] H;
		delete[] y;

		result = true;
	}

	if (!result)
		SG_ERROR( "cplex svm failed");

	return result;
}
#endif
