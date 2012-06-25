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
#include <shogun/lib/slep/q1/eppMatrix.h>
#include <shogun/mathematics/lapack.h>

namespace shogun
{

SGMatrix<double> slep_mt_lr(
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

	int n_tasks = options.n_nodes;

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
					b = m1[t]/(m1[t]+m2[t]);
				else
					b = -m2[t]/(m1[t]+m2[t]);

				features->add_to_dense_vec(b,i,ATb+t*n_feats,n_feats);
			}
			
		}

		double q_bar = 0.0;
		if (options.q==1)
			q_bar = CMath::ALMOST_INFTY;
		else if (options.q>1e-6)
			q_bar = 1;
		else
			q_bar = options.q/(options.q-1);
		lambda_max = 0.0;

		for (t=0; t<n_tasks; t++)
		{
			lambda_max = 
				CMath::max(lambda_max, 
						SGVector<float64_t>::qnorm(ATb+t*n_feats, n_feats, q_bar));
		}

		lambda_max /= n_vecs;

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
	double* c = SG_CALLOC(double, n_tasks);
	for (t=0; t<n_tasks; t++)
		c[t] = CMath::log(m1[t]/m2[t]);

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
	
	while (!done && iter < options.max_iter) 
	{
		beta = (alphap-1.0)/alpha;

		for (i=0; i<n_feats*n_tasks; i++)
			s[i] = w[i] + beta*wwp[i];
		for (t=0; t<n_tasks; t++)
			sc[t] = c[t] + beta*ccp[t];

		for (i=0; i<n_vecs; i++)
			As[i] = Aw[i] + beta*(Aw[i]-Awp[i]);

		double fun_s = 0.0;
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
	
		for (i=0; i<n_feats*n_tasks; i++)
			wp[i] = w[i];
		
		for (t=0; t<n_tasks; t++)
			cp[t] = c[t];

		for (i=0; i<n_vecs; i++)
			Awp[i] = Aw[i];

		while (true)
		{
			// v = s - g / L
			for (i=0; i<n_feats*n_tasks; i++)
				v[i] = s[i] - g[i]*(1.0/L);

			for (t=0; t<n_tasks; t++)
				c[t] = sc[t] - gc[t]*(1.0/L);

			eppMatrix(w.matrix, v, n_feats, n_tasks, lambda/L, options.q);

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

			double r_sum = SGVector<float64_t>::dot(v,v,n_feats*n_tasks);
			double l_sum = fun_x - fun_s - SGVector<float64_t>::dot(v,g,n_feats*n_tasks);

			for (t=0; t<n_tasks; t++)
			{
				r_sum += CMath::sq(c[t] - sc[t]);
				l_sum -= (c[t] - sc[t])*gc[t];
			}
			r_sum /= 2.0;

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
		
		for (t=0; t<n_tasks; t++)
			ccp[t] = c[t] - cp[t];

		double regularizer = 0.0;
		for (i=0; i<n_feats; i++)
		{
			double w_row_norm = 0.0;
			for (t=0; t<n_tasks; t++)
				w_row_norm += CMath::pow(w(i,t),options.q);
			regularizer += CMath::pow(w_row_norm,1.0/options.q);
		}

		funcp = func;
		func = fun_x + lambda*regularizer;
		SG_SPRINT("Obj = %f + %f * %f = %f \n",fun_x, lambda, regularizer, func);

		//if (gradient_break)
		//	break;

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

	return w;
};
};
