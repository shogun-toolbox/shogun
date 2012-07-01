/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2012 Sergey Lisitsyn
 * Copyright (C) 2010-2012 Jun Liu, Jieping Ye
 */

#include <shogun/lib/slep/slep_mt_lr.h>
#include <shogun/mathematics/Math.h>
#include <shogun/lib/slep/tree/altra.h>
#include <shogun/lib/slep/tree/general_altra.h>
#include <shogun/mathematics/lapack.h>

namespace shogun
{

slep_result_t slep_tree_mt_lr(
		CDotFeatures* features,
		double* y,
		double z,
		const slep_options& options)
{
	int i,j,t;
	int n_feats = features->get_dim_feature_space();
	int n_vecs = features->get_num_vectors();
	double lambda, beta;
	double funcp = 0.0, func = 0.0;

	int n_tasks = options.n_tasks;
	//SG_SPRINT("N tasks = %d \n", n_tasks);

	int iter = 1;
	bool done = false;
	bool gradient_break = false;

	int* m1 = SG_CALLOC(int, n_tasks);
	int* m2 = SG_CALLOC(int, n_tasks);
	for (t=0; t<n_tasks; t++)
	{
		int task_ind_start = options.ind[t];
		int task_ind_end = options.ind[t+1];
		for (i=task_ind_start; i<task_ind_end; i++)
		{
			if (y[i]>0)
				m1[t]++;
			else
				m2[t]++;
		}
	}
	
	double* ATb = SG_CALLOC(double, n_feats*n_tasks);

	if (options.regularization!=0)
	{
		if (z<0 || z>1)
			SG_SERROR("z is not in range [0,1]");

		for (t=0; t<n_tasks; t++)
		{
			int task_ind_start = options.ind[t];
			int task_ind_end = options.ind[t+1];
			double b = 0.0;
			for (i=task_ind_start; i<task_ind_end; i++)
			{
				if (y[i]>0)
					b = double(m1[t])/(m1[t]+m2[t]);
				else
					b = -double(m2[t])/(m1[t]+m2[t]);

				features->add_to_dense_vec(b,i,ATb+t*n_feats,n_feats);
			}
			
		}

		double lambda_max = 0.0;
		if (options.general)
			lambda_max = general_findLambdaMax_mt(ATb, n_feats, n_tasks, options.G, options.ind_t, options.n_nodes);
		else
			lambda_max = findLambdaMax_mt(ATb, n_feats, n_tasks, options.ind_t, options.n_nodes);

		lambda_max /= n_feats*n_tasks;
		
		lambda = z*lambda_max;
	}
	else 
		lambda = z;

	SGMatrix<double> w(n_feats,n_tasks);
	w.zero();
	if (options.initial_w)
	{
		for (j=0; j<n_tasks; j++)
			for (i=0; i<n_feats; i++)
				w(i,j) = options.initial_w[j*n_feats+i];
	}
	SGVector<double> c(n_tasks);
	c.zero();

	double* s = SG_CALLOC(double, n_feats*n_tasks);
	double* sc = SG_CALLOC(double, n_tasks);
	double* g = SG_CALLOC(double, n_feats*n_tasks);
	double* v = SG_CALLOC(double, n_feats*n_tasks);

	double* Aw = SG_CALLOC(double, n_vecs);
	for (t=0; t<n_tasks; t++)
	{
		int task_ind_start = options.ind[t];
		int task_ind_end = options.ind[t+1];
		for (i=task_ind_start; i<task_ind_end; i++)
			Aw[i] = features->dense_dot(i,w.matrix+t*n_feats,n_feats);
	}

	double* Av = SG_MALLOC(double, n_vecs);
	double* As = SG_MALLOC(double, n_vecs);

	double L = 1.0/n_vecs;

	double* wp = SG_CALLOC(double, n_feats*n_tasks);
	for (i=0; i<n_feats*n_tasks; i++)
		wp[i] = w[i];
	double* Awp = SG_MALLOC(double, n_vecs);
	for (i=0; i<n_vecs; i++)
		Awp[i] = Aw[i];
	double* wwp = SG_CALLOC(double, n_feats*n_tasks);

	double* cp = SG_MALLOC(double, n_tasks);
	for (t=0; t<n_tasks; t++)
		cp[t] = c[t];
	double* ccp = SG_CALLOC(double, n_tasks);

	double* w_row = SG_MALLOC(double, n_tasks);

	double* gc = SG_MALLOC(double, n_tasks);
	double* b = SG_MALLOC(double, n_vecs);
	
	double alphap = 0.0;
	double alpha = 1.0;

	double fun_x = 0.0;
	
	while (!done && iter <= options.max_iter) 
	{
		beta = (alphap-1.0)/alpha;
		//SG_SPRINT("beta = %f \n", beta);

		for (i=0; i<n_feats*n_tasks; i++)
			s[i] = w[i] + beta*wwp[i];
		for (t=0; t<n_tasks; t++)
			sc[t] = c[t] + beta*ccp[t];

		for (i=0; i<n_vecs; i++)
			As[i] = Aw[i] + beta*(Aw[i]-Awp[i]);

		//SG_SPRINT("As = %f\n",SGVector<float64_t>::dot(As,As,n_vecs));

		double fun_s = 0.0;
		for (i=0; i<n_tasks*n_feats; i++)
			g[i] = 0.0;
		for (t=0; t<n_tasks; t++)
		{
			int task_ind_start = options.ind[t];
			int task_ind_end = options.ind[t+1];

			gc[t] = 0.0;
			for (i=task_ind_start; i<task_ind_end; i++)
			{
				double aa = -y[i]*(As[i]+sc[t]);
				double bb = CMath::max(aa,0.0);

				fun_s += CMath::log(CMath::exp(-bb) + CMath::exp(aa-bb)) + bb;
				
				double prob = 1.0/(1.0+CMath::exp(aa));
				b[i] = -y[i]*(1.0-prob) / n_vecs;
				gc[t] += b[i];
				
				features->add_to_dense_vec(b[i],i,g+t*n_feats,n_feats);
			}
		}
		fun_s /= n_vecs;
		
		//SG_SPRINT("fun_s = %f\n", fun_s);

		//SG_SPRINT("g = %f\n", SGVector<float64_t>::dot(g,g,n_feats*n_tasks));
	
		for (i=0; i<n_feats*n_tasks; i++)
			wp[i] = w[i];
		
		for (t=0; t<n_tasks; t++)
			cp[t] = c[t];

		for (i=0; i<n_vecs; i++)
			Awp[i] = Aw[i];

		int inner_iter = 1;
		while (inner_iter < 100)
		{
			// v = s - g / L
			for (i=0; i<n_feats*n_tasks; i++)
				v[i] = s[i] - g[i]*(1.0/L);

			for (t=0; t<n_tasks; t++)
				c[t] = sc[t] - gc[t]*(1.0/L);
			

			if (options.general)
				general_altra_mt(w.matrix, v, n_feats, n_tasks, options.G, options.ind_t, options.n_nodes, lambda/L);
			else
				altra_mt(w.matrix, v, n_feats, n_tasks, options.ind_t, options.n_nodes, lambda/L);

			//SG_SPRINT("params [%d,%d,%f,%f]\n", n_feats, n_tasks, lambda/L, options.q);

			//w.display_matrix();
			//SG_SPRINT("w = %f \n", SGVector<float64_t>::dot(w.matrix,w.matrix,n_feats*n_tasks));

			// v = x - s
			for (i=0; i<n_feats*n_tasks; i++)
				v[i] = w[i] - s[i];

			fun_x = 0.0;
			for (t=0; t<n_tasks; t++)
			{
				int task_ind_start = options.ind[t];
				int task_ind_end = options.ind[t+1];
				for (i=task_ind_start; i<task_ind_end; i++)
				{
					Aw[i] = features->dense_dot(i,w.matrix+t*n_feats,n_feats);
					double aa = -y[i]*(Aw[i]+c[t]);
					double bb = CMath::max(aa,0.0);

					fun_x += CMath::log(CMath::exp(-bb) + CMath::exp(aa-bb)) + bb;
				}
			}
			fun_x /= n_vecs;

			//SG_SPRINT("Aw = %f\n", SGVector<float64_t>::dot(Aw,Aw,n_vecs));
			//c.display_vector();

			double r_sum = SGVector<float64_t>::dot(v,v,n_feats*n_tasks);
			double l_sum = fun_x - fun_s - SGVector<float64_t>::dot(v,g,n_feats*n_tasks);

			for (t=0; t<n_tasks; t++)
			{
				r_sum += CMath::sq(c[t] - sc[t]);
				l_sum -= (c[t] - sc[t])*gc[t];
			}
			r_sum /= 2.0;

			//SG_SPRINT("sums = [%f, %f, %f]\n", r_sum, l_sum, fun_x);

			if (r_sum <= 1e-20)
			{
				gradient_break = true;
				break;
			}

			if (l_sum <= r_sum*L)
				break;
			else 
				L = CMath::max(2*L, l_sum/r_sum);
			inner_iter++;
		}

		alphap = alpha;
		alpha = 0.5*(1+CMath::sqrt(4*alpha*alpha+1));

		for (i=0; i<n_feats*n_tasks; i++)
			wwp[i] = w[i] - wp[i];
		
		for (t=0; t<n_tasks; t++)
			ccp[t] = c[t] - cp[t];

		double regularizer = 0.0;
		for (i=0; i<n_feats; i++)
		{
			double tree_norm = 0.0;

			for (t=0; t<n_tasks; t++)
				w_row[t] = w(i,t);

			if (options.general)
				tree_norm = general_treeNorm(w_row, n_tasks, options.G, options.ind_t, options.n_nodes);
			else
				tree_norm = treeNorm(w_row, n_tasks, options.ind_t, options.n_nodes);

			regularizer += tree_norm;
		}

		funcp = func;
		func = fun_x + lambda*regularizer;
		//SG_SPRINT("Obj = %f + %f * %f = %f \n",fun_x, lambda, regularizer, func);

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

		iter++;
	}

	SG_FREE(wp);
	SG_FREE(wwp);
	SG_FREE(s);
	SG_FREE(sc);
	SG_FREE(cp);
	SG_FREE(ccp);
	SG_FREE(g);
	SG_FREE(v);
	SG_FREE(Aw);
	SG_FREE(Awp);
	SG_FREE(Av);
	SG_FREE(As);
	SG_FREE(w_row);
	SG_FREE(gc);
	SG_FREE(b);
	SG_FREE(m1);
	SG_FREE(m2);
	SG_FREE(ATb);

	return slep_result_t(w,c);
};
};
