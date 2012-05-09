/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2012 Sergey Lisitsyn
 * Copyright (C) 2010-2012 Jun Liu, Jieping Ye
 */

#include <shogun/lib/slep/slep_tree_lsr.h>
#include <shogun/mathematics/Math.h>
#include <shogun/lib/slep/tree/altra.h>
#include <shogun/lib/slep/tree/general_altra.h>

namespace shogun
{

double* slep_tree_lsr(
		CDotFeatures* features,
		double* y,
		double z,
		const slep_options& options)
{
	int i;
	int n_feats = features->get_dim_feature_space();
	int n_vecs = features->get_num_vectors();
	double lambda, lambda_max, beta;
	double funcp, func;

	int iter = 1;
	bool done = false;
	bool gradient_break = false;

	double* ATy = SG_CALLOC(double, n_feats);
	for (i=0; i<n_vecs; i++)
		features->add_to_dense_vec(y[i],i,ATy,n_feats);

	if (options.regularization!=0)
	{
		if (options.general)
			lambda_max = findLambdaMax(ATy, n_vecs, options.ind, options.n_nodes);
		else
			lambda_max = general_findLambdaMax(ATy, n_vecs, options.G, 
			                                   options.ind, options.n_nodes);
		lambda = z*lambda_max;
	}
	else 
		lambda = z;

	double* w = SG_CALLOC(double, n_feats);
	if (options.initial_w)
	{
		for (i=0; i<n_feats; i++)
			w[i] = options.initial_w[i];
	}
	else
	{
		for (i=0; i<n_feats; i++)
			w[i] = 0.0;
	}

	double* s = SG_CALLOC(double, n_feats);
	double* g = SG_CALLOC(double, n_feats);
	double* v = SG_CALLOC(double, n_feats);

	double* Aw = SG_CALLOC(double, n_vecs);
	features->dense_dot_range(Aw,0,n_vecs,NULL,w,n_feats,0.0);
	double* Av = SG_MALLOC(double, n_vecs);
	double* As = SG_MALLOC(double, n_vecs);
	double* ATAs = SG_MALLOC(double, n_feats);
	double* resid = SG_MALLOC(double, n_vecs);

	double L = 1.0;

	double* wp = SG_CALLOC(double, n_feats);
	for (i=0; i<n_feats; i++)
		wp[i] = w[i];
	double* Awp = SG_MALLOC(double, n_vecs);
	for (i=0; i<n_vecs; i++)
		Awp[i] = Aw[i];
	double* wwp = SG_CALLOC(double, n_feats);

	double alphap = 0.0;
	double alpha = 1.0;
	
	while (!done && iter < options.max_iter) 
	{
		//CMath::display_vector(w,n_feats,"w");

		beta = (alphap-1.0)/alpha;

		for (i=0; i<n_feats; i++)
			s[i] = w[i] + beta*wwp[i];


		for (i=0; i<n_vecs; i++)
			As[i] = Aw[i] + beta*(Aw[i]-Awp[i]);

		// ATAs = A'*As
		for (i=0; i<n_feats; i++)
			ATAs[i] = 0.0;
		for (i=0; i<n_vecs; i++)
			features->add_to_dense_vec(As[i],i,ATAs,n_feats);
		
		// g = ATAs - ATy
		for (i=0; i<n_feats; i++)
			g[i] = ATAs[i] - ATy[i];

		for (i=0; i<n_feats; i++)
			wp[i] = w[i];

		for (i=0; i<n_vecs; i++)
			Awp[i] = Aw[i];

		while (true)
		{
			// v = s - g / L
			for (i=0; i<n_feats; i++)
				v[i] = s[i] - g[i]*(1.0/L);

			if (options.general)
				general_altra(w, v, n_feats, options.G, options.ind, options.n_nodes, lambda/L);
			else
				altra(w, v, n_feats, options.ind, options.n_nodes, lambda/L);

			//CMath::display_vector(w,n_feats,"w_inner");
			//SG_SPRINT("\n");

			// v = x - s
			for (i=0; i<n_feats; i++)
				v[i] = w[i] - s[i];

			features->dense_dot_range(Aw,0,n_vecs,NULL,w,n_feats,0.0);

			for (i=0; i<n_vecs; i++)
				Av[i] = Aw[i] - As[i];

			double r_sum = CMath::dot(v,v,n_feats);
			double l_sum = CMath::dot(Av,Av,n_vecs);

			if (r_sum <= 1e-20)
			{
				gradient_break = true;
				break;
			}

			if (l_sum <= r_sum*L)
				break;
			else 
				L = CMath::max(2*L, l_sum/r_sum);

			SG_SPRINT("L=%.3f\n",L);
		}

		alphap = alpha;
		alpha = 0.5*(1+CMath::sqrt(4*alpha*alpha+1));

		for (i=0; i<n_feats; i++)
			wwp[i] = w[i] - wp[i];

		// Aw - y
		for (i=0; i<n_vecs; i++)
			resid[i] = Aw[i] - y[i];

		double tree_norm; 
		if (options.general)
			tree_norm = general_treeNorm(w,n_feats,options.G,
			                             options.ind,options.n_nodes);
		else
			tree_norm = treeNorm(w,n_feats,options.ind,options.n_nodes);

		funcp = func;
		func = 0.5*CMath::dot(resid,resid,n_vecs) + lambda*tree_norm;

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
					SG_SPRINT("STEP=%.9f\n",step);
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
				norm_wwp = CMath::sqrt(CMath::dot(wwp,wwp,n_feats));
				if (norm_wwp <= options.tolerance)
					done = true;
				break;
			case 4:
				norm_wp = CMath::sqrt(CMath::dot(wp,wp,n_feats));
				norm_wwp = CMath::sqrt(CMath::dot(wwp,wwp,n_feats));
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

	return w;
};
};
