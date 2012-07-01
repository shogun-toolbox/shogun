/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2012 Sergey Lisitsyn
 * Copyright (C) 2010-2012 Jun Liu, Jieping Ye
 */

#include <shogun/lib/slep/slep_tree_mt_lsr.h>
#include <shogun/mathematics/Math.h>
#include <shogun/lib/slep/tree/altra.h>
#include <shogun/lib/slep/tree/general_altra.h>
#include <shogun/mathematics/lapack.h>

namespace shogun
{

SGMatrix<double> slep_tree_mt_lsr(
		CDotFeatures* features,
		double* y,
		double z,
		const slep_options& options)
{
	int i,j,t;
	int n_feats = features->get_dim_feature_space();
	int n_vecs = features->get_num_vectors();
	double lambda, lambda_max, beta;
	double funcp = 0.0, func = 0.0;

	int n_tasks = options.n_tasks;

	int iter = 1;
	bool done = false;
	bool gradient_break = false;

	double* ATy = SG_CALLOC(double, n_feats*n_tasks);
	for (t=0; t<n_tasks; t++)
	{
		int task_ind_start = options.ind[t]+1;
		int task_ind_end = options.ind[t+1];
		for (i=task_ind_start; i<task_ind_end; i++)
			features->add_to_dense_vec(y[i],i,ATy+t*n_feats,n_feats);
	}

	if (options.regularization!=0)
	{
		if (options.general)
			lambda_max = general_findLambdaMax_mt(ATy, n_feats, n_tasks, options.G, options.ind_t, options.n_nodes);
		else
			lambda_max = findLambdaMax_mt(ATy, n_feats, n_tasks, options.ind_t, options.n_nodes);
		lambda = z*lambda_max;
	}
	else 
		lambda = z;

	SGMatrix<double> w(n_feats,n_tasks);
	if (options.initial_w)
	{
		for (j=0; j<n_tasks; j++)
			for (i=0; i<n_feats; i++)
				w(i,j) = options.initial_w[j*n_feats+i];
	}
	else
	{
		for (j=0; j<n_tasks*n_feats; j++)
				w[j] = 0.0;
	}

	double* s = SG_CALLOC(double, n_feats*n_tasks);
	double* g = SG_CALLOC(double, n_feats*n_tasks);
	double* v = SG_CALLOC(double, n_feats*n_tasks);

	double* Aw = SG_CALLOC(double, n_vecs);
	for (t=0; t<n_tasks; t++)
	{
		int task_ind_start = options.ind[t]+1;
		int task_ind_end = options.ind[t+1];
		for (i=task_ind_start; i<task_ind_end; i++)
			Aw[i] = features->dense_dot(i,w.matrix+t*n_feats,n_feats);
	}
	double* Av = SG_MALLOC(double, n_vecs);
	double* As = SG_MALLOC(double, n_vecs);
	double* ATAs = SG_MALLOC(double, n_feats*n_tasks);
	double* resid = SG_MALLOC(double, n_vecs*n_tasks);

	double L = 1.0;

	double* wp = SG_CALLOC(double, n_feats*n_tasks);
	for (i=0; i<n_feats*n_tasks; i++)
		wp[i] = w[i];
	double* Awp = SG_MALLOC(double, n_vecs);
	for (i=0; i<n_vecs; i++)
		Awp[i] = Aw[i];
	double* wwp = SG_CALLOC(double, n_feats*n_tasks);

	double* w_row = SG_MALLOC(double, n_tasks);

	double alphap = 0.0;
	double alpha = 1.0;
	
	while (!done && iter < options.max_iter) 
	{
		beta = (alphap-1.0)/alpha;

		for (i=0; i<n_feats*n_tasks; i++)
			s[i] = w[i] + beta*wwp[i];

		for (i=0; i<n_vecs; i++)
			As[i] = Aw[i] + beta*(Aw[i]-Awp[i]);

		// ATAs = A'*As
		for (i=0; i<n_feats*n_tasks; i++)
			ATAs[i] = 0.0;

		for (t=0; t<n_tasks; t++)
		{
			int task_ind_start = options.ind[t]+1;
			int task_ind_end = options.ind[t+1];
			for (i=task_ind_start; i<task_ind_end; i++)
				features->add_to_dense_vec(As[i],i,ATAs+t*n_feats,n_feats);
		}
		
		// g = ATAs - ATy
		for (i=0; i<n_feats*n_tasks; i++)
			g[i] = ATAs[i] - ATy[i];

		for (i=0; i<n_feats*n_tasks; i++)
			wp[i] = w[i];

		for (i=0; i<n_vecs; i++)
			Awp[i] = Aw[i];

		while (true)
		{
			// v = s - g / L
			for (i=0; i<n_feats*n_tasks; i++)
				v[i] = s[i] - g[i]*(1.0/L);

			if (options.general)
				general_altra_mt(w.matrix, v, n_feats, n_tasks, options.G, options.ind_t, options.n_nodes, lambda/L);
			else
				altra_mt(w.matrix, v, n_feats, n_tasks, options.ind_t, options.n_nodes, lambda/L);

			// v = x - s
			for (i=0; i<n_feats*n_tasks; i++)
				v[i] = w[i] - s[i];
	
			for (t=0; t<n_tasks; t++)
			{
				int task_ind_start = options.ind[t]+1;
				int task_ind_end = options.ind[t+1];
				for (i=task_ind_start; i<task_ind_end; i++)
					Aw[i] = features->dense_dot(i,w.matrix+t*n_feats,n_feats);
			}

			for (i=0; i<n_vecs; i++)
				Av[i] = Aw[i] - As[i];

			// squared frobenius norm of r
			double r_sum = SGVector<float64_t>::dot(v,v,n_feats*n_tasks);

			double l_sum = SGVector<float64_t>::dot(Av,Av,n_vecs);

			if (r_sum <= 1e-20)
			{
				gradient_break = true;
				break;
			}

			if (l_sum <= r_sum*L)
				break;
			else 
				L = CMath::max(2*L, l_sum/r_sum);
		}

		alphap = alpha;
		alpha = 0.5*(1+CMath::sqrt(4*alpha*alpha+1));

		for (i=0; i<n_feats*n_tasks; i++)
			wwp[i] = w[i] - wp[i];

		// Aw - y
		for (i=0; i<n_vecs; i++)
			resid[i] = Aw[i] - y[i];

		double tree_norm = 0.0; 
		for (i=0; i<n_feats; i++)
		{
			for (j=0; j<n_tasks; j++)
				w_row[j] = w(i,j);

			if (options.general)
				tree_norm += general_treeNorm(w_row,n_tasks,options.G,
											 options.ind_t,options.n_nodes);
			else
				tree_norm += treeNorm(w_row,n_tasks,options.ind_t,options.n_nodes);
		}

		funcp = func;
		func = 0.5*SGVector<float64_t>::dot(resid,resid,n_vecs) + lambda*tree_norm;

		if (gradient_break)
			break;

		double norm_wp, norm_wwp;
		double step;
		switch (options.termination)
		{
			case 0:
				if (iter>=2)
				{
					step = CMath::abs(func-funcp);

					if (step <= options.tolerance)
						done = true;
				}
				break;
			case 1:
				if (iter>=2)
				{
					step = CMath::abs(func-funcp);
					if (step <= step*options.tolerance)
						done = true;
				}
				break;
			case 2:
				if (func <= options.tolerance)
					done = true;
				break;
			case 3:
				norm_wwp = CMath::sqrt(SGVector<float64_t>::dot(wwp,wwp,n_feats));
				if (norm_wwp <= options.tolerance)
					done = true;
				break;
			case 4:
				norm_wp = CMath::sqrt(SGVector<float64_t>::dot(wp,wp,n_feats));
				norm_wwp = CMath::sqrt(SGVector<float64_t>::dot(wwp,wwp,n_feats));
				if (norm_wwp <= options.tolerance*CMath::max(norm_wp,1.0))
					done = true;
				break;
			case 5:
				if (iter > options.max_iter)
					done = true;
				break;
			default: 
				done = true;
		}

		if (iter%options.restart_num==0)
		{
			alphap = 0.0;
			alpha = 1.0;
			L = 0.5*L;
			for (i=0; i<n_feats; i++)
				wp[i] = w[i];

			for (i=0; i<n_vecs; i++)
				Awp[i] = Aw[i];

			for (i=0; i<n_feats; i++)
				wwp[i] = 0.0;
		}

		iter++;
	}

	SG_FREE(ATy);
	SG_FREE(wp);
	SG_FREE(wwp);
	SG_FREE(s);
	SG_FREE(g);
	SG_FREE(v);
	SG_FREE(Aw);
	SG_FREE(Awp);
	SG_FREE(Av);
	SG_FREE(As);
	SG_FREE(ATAs);
	SG_FREE(resid);
	SG_FREE(w_row);

	return w;
};
};
