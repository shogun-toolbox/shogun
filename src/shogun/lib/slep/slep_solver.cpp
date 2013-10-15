/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2012 Sergey Lisitsyn
 * Copyright (C) 2010-2012 Jun Liu, Jieping Ye
 */

#include <shogun/lib/slep/slep_solver.h>
#include <shogun/mathematics/Math.h>
#include <shogun/lib/slep/q1/eppMatrix.h>
#include <shogun/lib/slep/q1/eppVector.h>
#include <shogun/lib/slep/flsa/flsa.h>
#include <shogun/lib/slep/tree/altra.h>
#include <shogun/lib/slep/tree/general_altra.h>
#include <shogun/lib/Signal.h>

namespace shogun
{

double compute_regularizer(double* w, double lambda, double lambda2, int n_vecs, int n_feats,
                           int n_blocks, const slep_options& options)
{
	double regularizer = 0.0;
	switch (options.mode)
	{
		case MULTITASK_GROUP:
		{
			for (int i=0; i<n_feats; i++)
			{
				double w_row_norm = 0.0;
				for (int t=0; t<n_blocks; t++)
					w_row_norm += CMath::pow(w[i+t*n_feats],options.q);
				regularizer += CMath::pow(w_row_norm,1.0/options.q);
			}
			regularizer *= lambda;
		}
		break;
		case MULTITASK_TREE:
		{
			for (int i=0; i<n_feats; i++)
			{
				double tree_norm = 0.0;

				if (options.general)
					tree_norm = general_treeNorm(w+i, n_blocks, n_blocks, options.G, options.ind_t, options.n_nodes);
				else
					tree_norm = treeNorm(w+i, n_blocks, n_blocks, options.ind_t, options.n_nodes);

				regularizer += tree_norm;
			}
			regularizer *= lambda;
		}
		break;
		case FEATURE_GROUP:
		{
			for (int t=0; t<n_blocks; t++)
			{
				double group_qpow_sum = 0.0;
				int group_ind_start = options.ind[t];
				int group_ind_end = options.ind[t+1];
				for (int i=group_ind_start; i<group_ind_end; i++)
					group_qpow_sum += CMath::pow(w[i], options.q);

				regularizer += options.gWeight[t]*CMath::pow(group_qpow_sum, 1.0/options.q);
			}
			regularizer *= lambda;
		}
		break;
		case FEATURE_TREE:
		{
			if (options.general)
				regularizer = general_treeNorm(w, 1, n_feats, options.G, options.ind_t, options.n_nodes);
			else
				regularizer = treeNorm(w, 1, n_feats, options.ind_t, options.n_nodes);

			regularizer *= lambda;
		}
		break;
		case PLAIN:
		{
			for (int i=0; i<n_feats; i++)
				regularizer += CMath::abs(w[i]);

			regularizer *= lambda;
		}
		break;
		case FUSED:
		{
			double l1 = 0.0;
			for (int i=0; i<n_feats; i++)
				l1 += CMath::abs(w[i]);
			regularizer += lambda*l1;
			double fuse = 0.0;
			for (int i=1; i<n_feats; i++)
				fuse += CMath::abs(w[i]-w[i-1]);
			regularizer += lambda2*fuse;
		}
		break;
	}
	return regularizer;
};

double compute_lambda(
		double* ATx,
		double z,
		CDotFeatures* features,
		double* y,
		int n_vecs, int n_feats,
		int n_blocks,
		const slep_options& options)
{
	double lambda_max = 0.0;
	if (z<0 || z>1)
		SG_SERROR("z is not in range [0,1]")

	double q_bar = 0.0;
	if (options.q==1)
		q_bar = CMath::ALMOST_INFTY;
	else if (options.q>1e6)
		q_bar = 1;
	else
		q_bar = options.q/(options.q-1);

	SG_SINFO("q bar = %f \n",q_bar)

	switch (options.mode)
	{
		case MULTITASK_GROUP:
		case MULTITASK_TREE:
		{
			for (int t=0; t<n_blocks; t++)
			{
				SGVector<index_t> task_idx = options.tasks_indices[t];
				int n_vecs_task = task_idx.vlen;

				switch (options.loss)
				{
					case LOGISTIC:
					{
						double b = 0.0;
						int m1 = 0, m2 = 0;
						for (int i=0; i<n_vecs_task; i++)
						{
							if (y[task_idx[i]]>0)
								m1++;
							else
								m2++;
						}
						for (int i=0; i<n_vecs_task; i++)
						{
							if (y[task_idx[i]]>0)
								b = double(m1)/(m1+m2);
							else
								b = -double(m2)/(m1+m2);

							features->add_to_dense_vec(b,task_idx[i],ATx+t*n_feats,n_feats);
						}
					}
					break;
					case LEAST_SQUARES:
					{
						for (int i=0; i<n_vecs_task; i++)
							features->add_to_dense_vec(y[task_idx[i]],task_idx[i],ATx+t*n_feats,n_feats);
					}
				}
			}
		}
		break;
		case FEATURE_GROUP:
		case FEATURE_TREE:
		case PLAIN:
		case FUSED:
		{
			switch (options.loss)
			{
				case LOGISTIC:
				{
					int m1 = 0, m2 = 0;
					double b = 0.0;
					for (int i=0; i<n_vecs; i++)
						y[i]>0 ? m1++ : m2++;

					SG_SDEBUG("# pos = %d , # neg = %d\n",m1,m2)

					for (int i=0; i<n_vecs; i++)
					{
						y[i]>0 ? b=double(m2) / CMath::sq(n_vecs) : b=-double(m1) / CMath::sq(n_vecs);
						features->add_to_dense_vec(b,i,ATx,n_feats);
					}
				}
				break;
				case LEAST_SQUARES:
				{
					for (int i=0; i<n_vecs; i++)
						features->add_to_dense_vec(y[i],i,ATx,n_feats);
				}
				break;
			}
		}
		break;
	}

	switch (options.mode)
	{
		case MULTITASK_GROUP:
		{
			for (int i=0; i<n_feats; i++)
			{
				double sum = 0.0;
				for (int t=0; t<n_blocks; t++)
					sum += CMath::pow(fabs(ATx[t*n_feats+i]),q_bar);
				lambda_max =
					CMath::max(lambda_max, CMath::pow(sum,1.0/q_bar));
			}

			if (options.loss==LOGISTIC)
				lambda_max /= n_vecs;
		}
		break;
		case MULTITASK_TREE:
		{
			if (options.general)
				lambda_max = general_findLambdaMax_mt(ATx, n_feats, n_blocks, options.G, options.ind_t, options.n_nodes);
			else
				lambda_max = findLambdaMax_mt(ATx, n_feats, n_blocks, options.ind_t, options.n_nodes);

			lambda_max /= n_vecs*n_blocks;
		}
		break;
		case FEATURE_GROUP:
		{
			for (int t=0; t<n_blocks; t++)
			{
				int group_ind_start = options.ind[t];
				int group_ind_end = options.ind[t+1];
				double sum = 0.0;
				for (int i=group_ind_start; i<group_ind_end; i++)
					sum += CMath::pow(fabs(ATx[i]),q_bar);

				sum = CMath::pow(sum, 1.0/q_bar);
				sum /= options.gWeight[t];
				SG_SINFO("sum = %f\n",sum)
				if (sum>lambda_max)
					lambda_max = sum;
			}
		}
		break;
		case FEATURE_TREE:
		{
			if (options.general)
				lambda_max = general_findLambdaMax(ATx, n_feats, options.G, options.ind_t, options.n_nodes);
			else
				lambda_max = findLambdaMax(ATx, n_feats, options.ind_t, options.n_nodes);
		}
		break;
		case PLAIN:
		case FUSED:
		{
			double max = 0.0;
			for (int i=0; i<n_feats; i++)
			{
				if (CMath::abs(ATx[i]) > max)
					max = CMath::abs(ATx[i]);
			}
			lambda_max = max;
		}
		break;
	}

	SG_SINFO("Computed lambda = %f * %f = %f\n",z,lambda_max,z*lambda_max)
	return z*lambda_max;
}

void projection(double* w, double* v, int n_feats, int n_blocks, double lambda, double lambda2,
                double L, double* z, double* z0, const slep_options& options)
{
	switch (options.mode)
	{
		case MULTITASK_GROUP:
			eppMatrix(w, v, n_feats, n_blocks, lambda/L, options.q);
		break;
		case MULTITASK_TREE:
			if (options.general)
				general_altra_mt(w, v, n_feats, n_blocks, options.G, options.ind_t, options.n_nodes, lambda/L);
			else
				altra_mt(w, v, n_feats, n_blocks, options.ind_t, options.n_nodes, lambda/L);
		break;
		case FEATURE_GROUP:
			eppVector(w, v, options.ind, n_blocks, n_feats, options.gWeight, lambda/L, options.q > 1e6 ? 1e6 : options.q);
		break;
		case FEATURE_TREE:
			if (options.general)
				general_altra(w, v, n_feats, options.G, options.ind_t, options.n_nodes, lambda/L);
			else
				altra(w, v, n_feats, options.ind_t, options.n_nodes, lambda/L);
		break;
		case PLAIN:
			for (int i=0; i<n_feats; i++)
				w[i] = CMath::sign(v[i])*CMath::max(0.0,CMath::abs(v[i])-lambda/L);
		break;
		case FUSED:
			flsa(w,z,NULL,v,z0,lambda/L,lambda2/L,n_feats,1000,1e-8,1,6);
			for (int i=0; i<n_feats; i++)
				z0[i] = z[i];
		break;
	}

}

double search_point_gradient_and_objective(CDotFeatures* features, double* ATx, double* As,
                                           double* sc, double* y, int n_vecs,
                                           int n_feats, int n_tasks,
                                           double* g, double* gc,
                                           const slep_options& options)
{
	double fun_s = 0.0;
	//SG_SDEBUG("As=%f\n", SGVector<float64_t>::dot(As,As,n_vecs))
	//SG_SDEBUG("sc=%f\n", SGVector<float64_t>::dot(sc,sc,n_tasks))
	switch (options.mode)
	{
		case MULTITASK_GROUP:
		case MULTITASK_TREE:
			for (int t=0; t<n_tasks; t++)
			{
				SGVector<index_t> task_idx = options.tasks_indices[t];
				int n_vecs_task = task_idx.vlen;
				switch (options.loss)
				{
					case LOGISTIC:
						gc[t] = 0.0;
						for (int i=0; i<n_vecs_task; i++)
						{
							double aa = -y[task_idx[i]]*(As[task_idx[i]]+sc[t]);
							double bb = CMath::max(aa,0.0);
							fun_s += (CMath::log(CMath::exp(-bb) + CMath::exp(aa-bb)) + bb)/ n_vecs;
							double prob = 1.0/(1.0+CMath::exp(aa));
							double b = -y[task_idx[i]]*(1.0-prob) / n_vecs;
							gc[t] += b;
							features->add_to_dense_vec(b,task_idx[i],g+t*n_feats,n_feats);
						}
					break;
					case LEAST_SQUARES:
						for (int i=0; i<n_feats*n_tasks; i++)
							g[i] = -ATx[i];
						for (int i=0; i<n_vecs_task; i++)
							features->add_to_dense_vec(As[task_idx[i]],task_idx[i],g+t*n_feats,n_feats);
					break;
				}
			}
		break;
		case FEATURE_GROUP:
		case FEATURE_TREE:
		case PLAIN:
		case FUSED:
			switch (options.loss)
			{
				case LOGISTIC:
					gc[0] = 0.0;

					for (int i=0; i<n_vecs; i++)
					{
						double aa = -y[i]*(As[i]+sc[0]);
						double bb = CMath::max(aa,0.0);
						fun_s += (CMath::log(CMath::exp(-bb) + CMath::exp(aa-bb)) + bb);
						/*
						if (y[i]>0)
							fun_s += (CMath::log(CMath::exp(-bb) + CMath::exp(aa-bb)) + bb)*pos_weight;
						else
							fun_s += (CMath::log(CMath::exp(-bb) + CMath::exp(aa-bb)) + bb)*neg_weight;
						*/
						double prob = 1.0/(1.0+CMath::exp(aa));
						//double b = 0;
						double b = -y[i]*(1.0-prob)/n_vecs;
						/*
						if (y[i]>0)
							b = -y[i]*(1.0-prob)*pos_weight;
						else
							b = -y[i]*(1.0-prob)*neg_weight;
						*/
						gc[0] += b;
						features->add_to_dense_vec(b,i,g,n_feats);
					}
					fun_s /= n_vecs;
				break;
				case LEAST_SQUARES:
					for (int i=0; i<n_feats; i++)
						g[i] = -ATx[i];
					for (int i=0; i<n_vecs; i++)
						features->add_to_dense_vec(As[i],i,g,n_feats);
				break;
			}
		break;
	}
	SG_SDEBUG("G=%f\n", SGVector<float64_t>::dot(g,g,n_feats*n_tasks))

	return fun_s;
}

slep_result_t slep_solver(
		CDotFeatures* features,
		double* y,
		double z,
		const slep_options& options)
{
	int i,t;
	int n_feats = features->get_dim_feature_space();
	int n_vecs = features->get_num_vectors();
	double lambda, beta;
	double funcp = 0.0, func = 0.0;

	int n_blocks = 0;
	int n_tasks = 0;

	switch (options.mode)
	{
		case MULTITASK_GROUP:
		case MULTITASK_TREE:
			n_tasks = options.n_tasks;
			n_blocks = options.n_tasks;
		break;
		case FEATURE_GROUP:
		case FEATURE_TREE:
			n_tasks = 1;
			n_blocks = options.n_feature_blocks;
		break;
		case PLAIN:
		case FUSED:
			n_tasks = 1;
			n_blocks = 1;
		break;
	}
	SG_SDEBUG("n_tasks = %d, n_blocks = %d\n",n_tasks,n_blocks)
	SG_SDEBUG("n_nodes = %d\n",options.n_nodes)

	int iter = 1;
	bool done = false;
	bool gradient_break = false;

	double rsL2 = options.rsL2;

	double* ATx = SG_CALLOC(double, n_feats*n_tasks);
	if (options.regularization!=0)
	{
		lambda = compute_lambda(ATx, z, features, y, n_vecs, n_feats, n_blocks, options);
		rsL2*= lambda;
	}
	else
		lambda = z;

	double lambda2 = 0.0;

	SGMatrix<double> w(n_feats,n_tasks);
	w.zero();
	SGVector<double> c(n_tasks);
	c.zero();

	if (options.last_result)
	{
		w = options.last_result->w;
		c = options.last_result->c;
	}

	double* s = SG_CALLOC(double, n_feats*n_tasks);
	double* sc = SG_CALLOC(double, n_tasks);
	double* g = SG_CALLOC(double, n_feats*n_tasks);
	double* v = SG_CALLOC(double, n_feats*n_tasks);
	double* z_flsa = SG_CALLOC(double, n_feats);
	double* z0_flsa = SG_CALLOC(double, n_feats);

	double* Aw = SG_CALLOC(double, n_vecs);
	switch (options.mode)
	{
		case MULTITASK_GROUP:
		case MULTITASK_TREE:
		{
			for (t=0; t<n_blocks; t++)
			{
				SGVector<index_t> task_idx = options.tasks_indices[t];
				//task_idx.display_vector("task");
				int n_vecs_task = task_idx.vlen;
				for (i=0; i<n_vecs_task; i++)
					Aw[task_idx[i]] = features->dense_dot(task_idx[i],w.matrix+t*n_feats,n_feats);
			}
		}
		break;
		case FEATURE_GROUP:
		case FEATURE_TREE:
		case PLAIN:
		case FUSED:
		{
			for (i=0; i<n_vecs; i++)
				Aw[i] = features->dense_dot(i,w.matrix,n_feats);
		}
		break;
	}

	double* Av = SG_MALLOC(double, n_vecs);
	double* As = SG_MALLOC(double, n_vecs);

	double L = 1.0/n_vecs;

	if (options.mode==FUSED)
		L += rsL2;

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

	double* gc = SG_MALLOC(double, n_tasks);
	double alphap = 0.0, alpha = 1.0;
	double fun_x = 0.0;

	while (!done && iter <= options.max_iter && !CSignal::cancel_computations())
	{
		beta = (alphap-1.0)/alpha;

		for (i=0; i<n_feats*n_tasks; i++)
			s[i] = w[i] + beta*wwp[i];
		for (t=0; t<n_tasks; t++)
			sc[t] = c[t] + beta*ccp[t];
		for (i=0; i<n_vecs; i++)
			As[i] = Aw[i] + beta*(Aw[i]-Awp[i]);
		for (i=0; i<n_tasks*n_feats; i++)
			g[i] = 0.0;

		double fun_s = search_point_gradient_and_objective(features, ATx, As, sc, y, n_vecs, n_feats, n_tasks, g, gc, options);

		//SG_SDEBUG("fun_s = %f\n", fun_s)

		if (options.mode==PLAIN || options.mode==FUSED)
			fun_s += rsL2/2 * SGVector<float64_t>::dot(w.matrix,w.matrix,n_feats);

		for (i=0; i<n_feats*n_tasks; i++)
			wp[i] = w[i];
		for (t=0; t<n_tasks; t++)
			cp[t] = c[t];
		for (i=0; i<n_vecs; i++)
			Awp[i] = Aw[i];

		int inner_iter = 1;
		while (inner_iter <= 1000)
		{
			for (i=0; i<n_feats*n_tasks; i++)
				v[i] = s[i] - g[i]*(1.0/L);

			for (t=0; t<n_tasks; t++)
				c[t] = sc[t] - gc[t]*(1.0/L);

			projection(w.matrix,v,n_feats,n_blocks,lambda,lambda2,L,z_flsa,z0_flsa,options);

			for (i=0; i<n_feats*n_tasks; i++)
				v[i] = w[i] - s[i];

			fun_x = 0.0;
			switch (options.mode)
			{
				case MULTITASK_GROUP:
				case MULTITASK_TREE:
					for (t=0; t<n_blocks; t++)
					{
						SGVector<index_t> task_idx = options.tasks_indices[t];
						int n_vecs_task = task_idx.vlen;
						for (i=0; i<n_vecs_task; i++)
						{
							Aw[task_idx[i]] = features->dense_dot(task_idx[i],w.matrix+t*n_feats,n_feats);
							if (options.loss==LOGISTIC)
							{
								double aa = -y[task_idx[i]]*(Aw[task_idx[i]]+c[t]);
								double bb = CMath::max(aa,0.0);
								fun_x += (CMath::log(CMath::exp(-bb) + CMath::exp(aa-bb)) + bb);
							}
						}
					}
				break;
				case FEATURE_GROUP:
				case FEATURE_TREE:
				case PLAIN:
				case FUSED:
					for (i=0; i<n_vecs; i++)
					{
						Aw[i] = features->dense_dot(i, w.matrix, n_feats);
						if (options.loss==LOGISTIC)
						{
							double aa = -y[i]*(Aw[i]+c[0]);
							double bb = CMath::max(aa,0.0);
							if (y[i]>0)
								fun_x += (CMath::log(CMath::exp(-bb) + CMath::exp(aa-bb)) + bb);//*pos_weight;
							else
								fun_x += (CMath::log(CMath::exp(-bb) + CMath::exp(aa-bb)) + bb);//*neg_weight;
						}
					}
				break;
			}
			if (options.loss==LOGISTIC)
				fun_x /= n_vecs;
			if (options.mode==PLAIN || options.mode==FUSED)
				fun_x += rsL2/2 * SGVector<float64_t>::dot(w.matrix,w.matrix,n_feats);

			double l_sum = 0.0, r_sum = 0.0;
			switch (options.loss)
			{
				case LOGISTIC:
					r_sum = SGVector<float64_t>::dot(v,v,n_feats*n_tasks);
					l_sum = fun_x - fun_s - SGVector<float64_t>::dot(v,g,n_feats*n_tasks);
					for (t=0; t<n_tasks; t++)
					{
						r_sum += CMath::sq(c[t] - sc[t]);
						l_sum -= (c[t] - sc[t])*gc[t];
					}
					r_sum /= 2.0;
				break;
				case LEAST_SQUARES:
					r_sum = SGVector<float64_t>::dot(v,v,n_feats*n_tasks);
					for (i=0; i<n_vecs; i++)
						l_sum += CMath::sq(Aw[i]-As[i]);
				break;
			}

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
		double regularizer = compute_regularizer(w.matrix, lambda, lambda2, n_vecs, n_feats, n_blocks, options);
		funcp = func;

		if (options.loss==LOGISTIC)
		{
			func = fun_x + regularizer;
		}
		if (options.loss==LEAST_SQUARES)
		{
			func = regularizer;
			for (i=0; i<n_vecs; i++)
				func += CMath::sq(Aw[i] - y[i]);
		}
		SG_SDEBUG("Obj = %f + %f = %f \n",fun_x, regularizer, func)

		if (gradient_break)
		{
			SG_SINFO("Gradient norm is less than 1e-20\n")
			break;
		}

		double norm_wp, norm_wwp;
		double step;
		switch (options.termination)
		{
			case 0:
				if (iter>=2)
				{
					step = CMath::abs(func-funcp);
					if (step <= options.tolerance)
					{
						SG_SINFO("Objective changes less than tolerance\n")
						done = true;
					}
				}
			break;
			case 1:
				if (iter>=2)
				{
					step = CMath::abs(func-funcp);
					if (step <= step*options.tolerance)
					{
						SG_SINFO("Objective changes relatively less than tolerance\n")
						done = true;
					}
				}
			break;
			case 2:
				if (func <= options.tolerance)
				{
					SG_SINFO("Objective is less than tolerance\n")
					done = true;
				}
			break;
			case 3:
				norm_wwp = CMath::sqrt(SGVector<float64_t>::dot(wwp,wwp,n_feats*n_tasks));
				if (norm_wwp <= options.tolerance)
					done = true;
			break;
			case 4:
				norm_wp = CMath::sqrt(SGVector<float64_t>::dot(wp,wp,n_feats*n_tasks));
				norm_wwp = CMath::sqrt(SGVector<float64_t>::dot(wwp,wwp,n_feats*n_tasks));
				if (norm_wwp <= options.tolerance*CMath::max(norm_wp,1.0))
					done = true;
			break;
			default:
				done = true;
		}

		iter++;
	}
	SG_SINFO("Finished %d iterations, objective = %f\n", iter, func)

	SG_FREE(ATx);
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
	SG_FREE(gc);
	SG_FREE(z_flsa);
	SG_FREE(z0_flsa);

	return slep_result_t(w,c);
};
};
