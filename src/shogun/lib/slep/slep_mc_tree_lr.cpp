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
#include <shogun/lib/slep/tree/altra.h>
#include <shogun/lib/slep/q1/eppMatrix.h>
#include <shogun/mathematics/Math.h>
#include <shogun/mathematics/eigen3.h>
#include <shogun/lib/Signal.h>
#include <shogun/lib/Time.h>
#include <iostream>

using namespace shogun;
using namespace Eigen;
using namespace std;

namespace shogun
{

slep_result_t slep_mc_tree_lr(
		CDotFeatures* features,
		CMulticlassLabels* labels,
		float64_t z,
		const slep_options& options)
{
	int i,j;
	// obtain problem parameters
	int n_feats   = features->get_dim_feature_space();
	int n_vecs    = features->get_num_vectors();
	int n_classes = labels->get_num_classes();

	// labels vector containing values in range (0 .. n_classes)
	SGVector<float64_t> labels_vector = labels->get_labels();

	// initialize matrices and vectors to be used
	// weight vector
	MatrixXd w  = MatrixXd::Zero(n_feats, n_classes);
	// intercepts (biases)
	VectorXd c  = VectorXd::Zero(n_classes);

	if (options.last_result)
	{
		SGMatrix<float64_t> last_w = options.last_result->w;
		SGVector<float64_t> last_c = options.last_result->c;
		for (i=0; i<n_classes; i++)
		{
			c[i] = last_c[i];
			for (j=0; j<n_feats; j++)
				w(j,i) = last_w(j,i);
		}
	}
	// iterative process matrices and vectors
	MatrixXd wp = w, wwp = MatrixXd::Zero(n_feats, n_classes);
	VectorXd cp = c, ccp = VectorXd::Zero(n_classes);
	// search point weight vector
	MatrixXd search_w = MatrixXd::Zero(n_feats, n_classes);
	// search point intercepts
	VectorXd search_c = VectorXd::Zero(n_classes);
	// dot products
	MatrixXd Aw  = MatrixXd::Zero(n_vecs, n_classes);
	for (j=0; j<n_classes; j++)
		features->dense_dot_range(Aw.col(j).data(), 0, n_vecs, NULL, w.col(j).data(), n_feats, 0.0);
	MatrixXd As  = MatrixXd::Zero(n_vecs, n_classes);
	MatrixXd Awp = MatrixXd::Zero(n_vecs, n_classes);
	// gradients
	MatrixXd g   = MatrixXd::Zero(n_feats, n_classes);
	VectorXd gc  = VectorXd::Zero(n_classes);
	// projection
	MatrixXd v   = MatrixXd::Zero(n_feats, n_classes);

	// Lipschitz continuous gradient parameter for line search
	double L = 1.0/(n_vecs*n_classes);
	// coefficients for search point computation
	double alphap = 0, alpha = 1;

	// lambda regularization parameter
	double lambda = z;
	// objective values
	double objective = 0.0;
	double objective_p = 0.0;

	int iter = 0;
	bool done = false;
	CTime time;
	//internal::set_is_malloc_allowed(false);
	while ((!done) && (iter<options.max_iter) && (!CSignal::cancel_computations()))
	{
		double beta = (alphap-1)/alpha;
		// compute search points
		search_w = w + beta*wwp;
		search_c = c + beta*ccp;

		// update dot products with search point
		As = Aw + beta*(Aw-Awp);

		// compute objective and gradient at search point
		double fun_s = 0;
		g.setZero();
		gc.setZero();
		// for each vector
		for (i=0; i<n_vecs; i++)
		{
			// class of current vector
			int vec_class = labels_vector[i];
			// for each class
			for (j=0; j<n_classes; j++)
			{
				// compute logistic loss
				double aa = ((vec_class == j) ? -1.0 : 1.0)*(As(i,j) + search_c(j));
				double bb = aa > 0.0 ? aa : 0.0;
				// avoid underflow via log-sum-exp trick
				fun_s += CMath::log(CMath::exp(-bb) + CMath::exp(aa-bb)) + bb;
				double prob = 1.0/(1+CMath::exp(aa));
				double b = ((vec_class == j) ? -1.0 : 1.0)*(1-prob);///(n_vecs*n_classes);
				// update gradient of intercepts
				gc[j] += b;
				// update gradient of weight vectors
				features->add_to_dense_vec(b, i, g.col(j).data(), n_feats);
			}
		}
		//fun_s /= (n_vecs*n_classes);

		wp = w;
		Awp = Aw;
		cp = c;

		int inner_iter = 0;
		double fun_x = 0;

		// line search process
		while (inner_iter<5000)
		{
			// compute line search point
			v = search_w - g/L;
			c = search_c - gc/L;

			// compute projection of gradient
			if (options.general)
				general_altra_mt(w.data(),v.data(),n_classes,n_feats,options.G,options.ind_t,options.n_nodes,lambda/L);
			else
				altra_mt(w.data(),v.data(),n_classes,n_feats,options.ind_t,options.n_nodes,lambda/L);
			v = w - search_w;

			// update dot products
			for (j=0; j<n_classes; j++)
				features->dense_dot_range(Aw.col(j).data(), 0, n_vecs, NULL, w.col(j).data(), n_feats, 0.0);

			// compute objective at search point
			fun_x = 0;
			for (i=0; i<n_vecs; i++)
			{
				int vec_class = labels_vector[i];
				for (j=0; j<n_classes; j++)
				{
					double aa = ((vec_class == j) ? -1.0 : 1.0)*(Aw(i,j) + c(j));
					double bb = aa > 0.0 ? aa : 0.0;
					fun_x += CMath::log(CMath::exp(-bb) + CMath::exp(aa-bb)) + bb;
				}
			}
			//fun_x /= (n_vecs*n_classes);

			// check for termination of line search
			double r_sum = (v.squaredNorm() + (c-search_c).squaredNorm())/2;
			double l_sum = fun_x - fun_s - v.cwiseProduct(g).sum() - (c-search_c).dot(gc);

			// stop if projected gradient is less than 1e-20
			if (r_sum <= 1e-20)
			{
				SG_SINFO("Gradient step makes little improvement (%f)\n",r_sum)
				done = true;
				break;
			}

			if (l_sum <= r_sum*L)
				break;
			else
				L = CMath::max(2*L, l_sum/r_sum);

			inner_iter++;
		}

		// update alpha coefficients
		alphap = alpha;
		alpha = (1+CMath::sqrt(4*alpha*alpha+1))/2;

		// update wwp and ccp
		wwp = w - wp;
		ccp = c - cp;

		// update objectives
		objective_p = objective;
		objective = fun_x;

		// compute tree norm
		double tree_norm = 0.0;
		if (options.general)
		{
			for (i=0; i<n_classes; i++)
				tree_norm += general_treeNorm(w.col(i).data(),n_classes,n_feats,options.G,options.ind_t,options.n_nodes);
		}
		else
		{
			for (i=0; i<n_classes; i++)
				tree_norm += treeNorm(w.col(i).data(),n_classes,n_feats,options.ind_t,options.n_nodes);
		}

		// regularize objective with tree norm
		objective += lambda*tree_norm;

		//cout << "Objective = " << objective << endl;

		// check for termination of whole process
		if ((CMath::abs(objective - objective_p) < options.tolerance*CMath::abs(objective_p)) && (iter>2))
		{
			SG_SINFO("Objective changes less than tolerance\n")
			done = true;
		}

		iter++;
	}
	SG_SINFO("%d iterations passed, objective = %f\n",iter,objective)
	//internal::set_is_malloc_allowed(true);

	// output computed weight vectors and intercepts
	SGMatrix<float64_t> r_w(n_feats,n_classes);
	for (j=0; j<n_classes; j++)
	{
		for (i=0; i<n_feats; i++)
			r_w(i,j) = w(i,j);
	}
	//r_w.display_matrix();
	SGVector<float64_t> r_c(n_classes);
	for (j=0; j<n_classes; j++)
		r_c[j] = c[j];
	return slep_result_t(r_w, r_c);
};
};
