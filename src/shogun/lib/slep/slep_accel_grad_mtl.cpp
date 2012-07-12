/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2012 Sergey Lisitsyn
 * Copyright (C) 2010-2012 Jun Liu, Jieping Ye
 */

#include <shogun/lib/slep/slep_accel_grad_mtl.h>
#include <shogun/features/DotFeatures.h>
#include <shogun/io/SGIO.h>
#include <shogun/mathematics/Math.h>
#include <shogun/mathematics/lapack.h>

using namespace shogun;

void compute_QP(double* Wp, double& P, double& sval,
		double* W1, double* delta_W, CDotFeatures* features,
		double* Y, double* W, int n_vecs, int n_feats, int* ind, int n_tasks, 
		double L, double lambda)
{
	int i,t;
	for (t=0; t<n_tasks; t++)
	{
		int task_ind_start = ind[t];
		int task_ind_end = ind[t+1];
		for (i=task_ind_start; i<task_ind_end; i++)
		{
			double resid = features->dense_dot(i,W+t*n_feats,n_feats) - Y[i];
			features->add_to_dense_vec(resid,i,delta_W+t*n_feats,n_feats);
		}
	}
	for (i=0; i<n_feats*n_tasks; i++)
		W1[i] = W[i] - (1.0/L)*delta_W[i];

	// assume n_tasks is less than n_feats
	ASSERT(n_tasks < n_feats);
	double* sing = SG_MALLOC(double, n_tasks);
	double* u = SG_MALLOC(double, n_feats*n_tasks);
	double* vt = SG_MALLOC(double, n_tasks*n_tasks);

	// follow with low-rank approximation using positive eigenpairs
	int info = 0;
	wrap_dgesvd('S','S',n_feats,n_tasks,W1,n_feats,sing,u,n_feats,vt,n_tasks,&info);

	for (i=0; i<n_tasks; i++)
	{
		if (sing[i] > 0)
			sval += sing[i] - lambda/L;

		cblas_dger(CblasColMajor, n_feats, n_tasks, sing[i] - lambda/L, u+i*n_feats, 1, vt+i*n_tasks, 1, Wp, n_feats);
	}

	// compute residual
	for (t=0; t<n_tasks; t++)
	{
		int task_ind_start = ind[t];
		int task_ind_end = ind[t+1];
		for (i=task_ind_start; i<task_ind_end; i++)
			P += 0.5*features->dense_dot(i,W+t*n_feats,n_feats) - Y[i];
	}

	// squared frobenius norm of W-Wp
	for (i=0; i<n_tasks*n_feats; i++)
		P += 0.5*L*CMath::sq(W[i]-Wp[i]);

	// trace(delta_W'*(W-Wp))
	for (t=0; t<n_tasks; t++)
	{
		for (i=0; i<n_feats; i++)
			P += delta_W[t*n_feats+i]*(W[t*n_feats+i]-Wp[t*n_feats+i]);
	}
}

namespace shogun
{

SGMatrix<double> slep_accel_grad_mtl(
		CDotFeatures* features,
		double* y,
		double lambda,
		const slep_options& options)
{
	SG_SWARNING("SLEP trace norm regularized MTL is not fully tested yet.\n");
	int i,t;
	int n_feats = features->get_dim_feature_space();
	int n_vecs = features->get_num_vectors();
	int n_tasks = options.n_tasks;

	double fval = 0.0;
	double fval_old = 1.0;

	double alpha = 1.0;
	double alpha_old = 1.0;

	double Q,P;
	double sval;
	double L = 100;
	double gamma = 1.1;

	double* Wp = SG_CALLOC(double, n_feats*n_tasks);
	double* W1 = SG_CALLOC(double, n_feats*n_tasks);
	double* delta_W = SG_CALLOC(double, n_feats*n_tasks);
	double* W_old = SG_CALLOC(double, n_feats*n_tasks);
	double* Z_old = SG_CALLOC(double, n_feats*n_tasks);

	int iter = 0;
	while (CMath::abs(fval_old-fval)/fval_old>options.tolerance)
	{
		fval_old = fval;
		compute_QP(Wp,P,sval,
		           W1,delta_W,features,y,Z_old,
		           n_vecs,n_feats,options.ind,n_tasks,
		           L,lambda);
		double f = 0.0;
		for (t=0; t<n_tasks; t++)
		{
			int task_ind_start = options.ind[t];
			int task_ind_end = options.ind[t+1];
			for (i=task_ind_start; i<task_ind_end; i++)
				f += CMath::sq(y[i]-features->dense_dot(i, Wp+t*n_feats, n_feats));
		}
		fval = f + lambda*sval;
		Q = P + lambda*sval;
		while (fval>Q)
		{
			L = L*gamma;
			compute_QP(Wp,P,sval,
			           W1,delta_W,features,y,Z_old,
			           n_vecs,n_feats,options.ind,n_tasks,
			           L,lambda);
			f = 0.0;
			for (t=0; t<n_tasks; t++)
			{
				int task_ind_start = options.ind[t];
				int task_ind_end = options.ind[t+1];
				for (i=task_ind_start; i<task_ind_end; i++)
					f += CMath::sq(y[i]-features->dense_dot(i, Wp+t*n_feats, n_feats));
			}
			
			fval = f + lambda*sval;
			Q = P + lambda*sval;
		}

		alpha_old = alpha;
		alpha = (1+CMath::sqrt(1+4*alpha_old*alpha_old))/2;

		for (i=0; i<n_feats*n_tasks; i++)
			Z_old[i] = Wp[i] + ((alpha_old-1)/alpha)*(Wp[i]-W_old[i]);

		for (i=0; i<n_feats*n_tasks; i++)
			W_old[i] = Wp[i];

		iter++;
		if (iter>options.max_iter)
			break;
	}
	
	SG_FREE(W1);
	SG_FREE(delta_W);
	SG_FREE(W_old);
	SG_FREE(Z_old);

	return SGMatrix<float64_t>(Wp,n_feats,n_tasks);
};
};
