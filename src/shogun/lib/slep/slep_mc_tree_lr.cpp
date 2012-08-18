/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2012 Sergey Lisitsyn
 * Copyright (C) 2010-2012 Jun Liu, Jieping Ye
 */

#include <shogun/lib/slep/slep_mc_tree_lr.h>
#include <shogun/lib/slep/tree/general_altra.h>
#include <shogun/mathematics/Math.h>
#include <shogun/mathematics/eigen3.h>
#include <shogun/lib/Signal.h>

using namespace shogun;
using namespace Eigen;
using namespace std;

slep_result_t slep_mc_tree_lr(
		CDotFeatures* features,
		CMulticlassLabels* labels,
		float64_t z,
		const slep_options& options)
{
	int i,j;
	int n_feats   = features->get_dim_feature_space();
	int n_vecs    = features->get_num_vectors();
	int n_classes = labels->get_num_classes();

	MatrixXd w  = MatrixXd::Zero(n_feats, n_classes);
	VectorXd c  = VectorXd::Zero(n_classes);
	MatrixXd wp = w, wwp = MatrixXd::Zero(n_feats, n_classes);
	VectorXd cp = c, ccp = VectorXd::Zero(n_classes);
	MatrixXd search_w = MatrixXd::Zero(n_feats, n_classes);
	VectorXd search_c = VectorXd::Zero(n_classes);
	
	VectorXd Aw  = VectorXd::Zero(n_vecs);
	VectorXd As  = VectorXd::Zero(n_vecs);
	VectorXd Awp = VectorXd::Zero(n_vecs);

	MatrixXd g   = MatrixXd::Zero(n_feats, n_classes);
	VectorXd gc  = VectorXd::Zero(n_classes);
	MatrixXd v   = MatrixXd::Zero(n_feats, n_classes);

	double L = 1.0/n_vecs;
	double alphap = 0, alpha = 1;

	double lambda = z;
	double objective = 0.0;
	double objective_p = 0.0;

	int iter = 0;
	bool done = false;
	while ((!done) && (iter<options.max_iter) && (!CSignal::cancel_computations()))
	{
		double beta = (alphap-1)/alpha;
		search_w = w + beta*wwp;
		search_c = c + beta*ccp;

		As = Aw + beta*(Aw-Awp);
		double fun_s = 0;
		g.setZero();
		gc.setZero();
		for (i=0; i<n_vecs; i++)
		{
			int vec_class = labels->get_label(i);
			for (j=0; j<n_classes; j++)
			{
				double aa = ((vec_class == j) ? -1.0 : 1.0)*(As(i) + search_c(i));
				double bb = aa > 0.0 ? aa : 0.0;
				fun_s += CMath::log(CMath::exp(-bb) + CMath::exp(aa-bb)) + bb;
				double prob = 1.0/(1+CMath::exp(aa));
				double b = ((vec_class == j) ? -1.0 : 1.0)*(1-prob)/n_vecs;
				gc[vec_class] += b;
				features->add_to_dense_vec(b, i, g.col(vec_class).data(), n_feats);
			}
		}
		fun_s /= n_vecs;
		
		wp = w;
		Awp = Aw;
		cp = c;

		int inner_iter = 0;
		double fun_x = 0;
		while (inner_iter<1000)
		{
			v = search_w - g/L;
			c = search_c - gc/L;

			general_altra_mt(w.data(),v.data(),n_classes,n_feats,options.G,options.ind_t,options.n_nodes,lambda/L);
			
			v = w - search_w;

			fun_x = 0;
			for (i=0; i<n_vecs; i++)
			{
				int vec_class = labels->get_label(i);
				features->dense_dot(i, w.col(vec_class).data(), n_feats);
				for (j=0; j<n_classes; j++)
				{
					double aa = ((vec_class == j) ? -1.0 : 1.0)*(As(i) + search_c(i));
					double bb = aa > 0.0 ? aa : 0.0;
					fun_x += CMath::log(CMath::exp(-bb) + CMath::exp(aa-bb)) + bb;
				}
			}
			fun_x /= n_vecs;

			double r_sum = (v.squaredNorm() + (c-search_c).squaredNorm())/2;
			double l_sum = fun_x - fun_s - v.cwiseProduct(g).sum() - (c-search_c).dot(gc);

			if (r_sum <= 1e-20)
			{
				SG_SINFO("Gradient step makes little improvement\n");
				done = true;
				break;
			}

			if (l_sum <= r_sum*L)
				break;
			else
				L = CMath::max(2*L, l_sum/r_sum);

			inner_iter++;
		}

		alphap = alpha;
		alpha = (1+CMath::sqrt(4*alpha*alpha+1))/2;

		wwp = w - wp;
		ccp = w - cp;

		objective_p = objective;
		objective = fun_x;

		double tree_norm = 0.0;
		for (i=0; i<n_classes; i++)
			tree_norm += general_treeNorm(w.col(i).data(),n_feats,1,options.G,options.ind_t,options.n_nodes);

		objective += lambda*tree_norm;

		if ((CMath::abs(objective - objective_p) < options.tolerance) && (iter>2))
		{
			SG_SINFO("Objective changes less than tolerance\n");
			done = true;
		}

		iter++;
	}

	SGMatrix<float64_t> r_w(n_feats,n_classes);
	for (j=0; j<n_classes; j++)
	{
		for (i=0; i<n_feats; i++)
			r_w(i,j) = w(i,j);
	}
	SGVector<float64_t> r_c(n_classes);
	for (j=0; j<n_classes; j++)
		r_c[j] = c[j];
	return slep_result_t(r_w, r_c);
}
