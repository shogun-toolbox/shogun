/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2012 Sergey Lisitsyn
 * Copyright (C) 2012 Jiayu Zhou and Jieping Ye
 */

#include <shogun/lib/slep/malsar_joint_feature_learning.h>
#include <shogun/mathematics/Math.h>
#include <iostream>

#define EIGEN_RUNTIME_NO_MALLOC
#define NDEBUG
#include <eigen3/Eigen/Dense>

using namespace Eigen;

namespace shogun
{

slep_result_t malsar_joint_feature_learning(
		CDotFeatures* features,
		double* y,
		double rho1,
		double rho2,
		const slep_options& options)
{
	int task;
	int n_feats = features->get_dim_feature_space();
	SG_SPRINT("n feats = %d\n", n_feats);
	int n_vecs = features->get_num_vectors();
	SG_SPRINT("n vecs = %d\n", n_vecs);
	int n_tasks = options.n_tasks;
	SG_SPRINT("n tasks = %d\n", n_tasks);

	int iter = 0;

	// initialize weight vector and bias for each task
	MatrixXd Ws = MatrixXd::Zero(n_feats, n_tasks);
	VectorXd Cs = VectorXd::Zero(n_tasks);
	for (task=0; task<n_tasks; task++)
	{
		int n_pos = 0;
		int n_neg = 0;
		for (int i=options.ind[task]; i<options.ind[task+1]; i++)
		{
			if (y[i] > 0)
				n_pos++;
			else
				n_neg++;
		}
		Cs[task] = CMath::log(double(n_pos)/n_neg);
	}

	MatrixXd Wz=Ws, Wzp=Ws, Wz_old=Ws, delta_Wzp=Ws, gWs=Ws;
	VectorXd Cz=Cs, Czp=Cs, Cz_old=Cs, delta_Czp=Cs, gCs=Cs;

	double t=1, t_old=0;
	double gamma=1, gamma_inc=2;
	double obj=0.0, obj_old=0.0;

	internal::set_is_malloc_allowed(false);
	while (iter < options.max_iter)
	{
		double alpha = double(t_old - 1)/t;

		// compute search point
		Ws = (1+alpha)*Wz - alpha*Wz_old;
		Cs = (1+alpha)*Cz - alpha*Cz_old;

		// zero gradient
		gWs.setZero();
		gCs.setZero();

		// compute gradient and objective at search point
		double Fs = 0;
		for (task=0; task<n_tasks; task++)
		{
			for (int i=options.ind[task]; i<options.ind[task+1]; i++)
			{
				double aa = -y[i]*(features->dense_dot(i, Ws.col(task).data(), n_feats)+Cs[task]);
				double bb = CMath::max(aa,0.0);

				// avoid underflow when computing exponential loss
				Fs += (CMath::log(CMath::exp(-bb) + CMath::exp(aa-bb)) + bb)/n_vecs;
				double b = -y[i]*(1 - 1/(1+CMath::exp(aa)))/n_vecs;

				gCs[task] += b;
				features->add_to_dense_vec(b, i, gWs.col(task).data(), n_feats);
			}
		}
		gWs.noalias() += 2*rho2*Ws;
		
		// add regularizer
		Fs += Ws.squaredNorm();

		double Fzp = 0.0;
		double gradient_break = false;

		// line search, Armijo-Goldstein scheme
		while (true)
		{
			// compute lasso projection of Ws - gWs/gamma
			for (task=0; task<n_tasks; task++)
			{
				Wzp.col(task) = Ws.col(task) - gWs.col(task)/gamma;
				double norm = Wzp.col(task).lpNorm<2>();
				Wzp.col(task) *= CMath::max(0.0,norm-rho1/gamma)/norm;
			}
			// walk in direction of antigradient 
			Czp = Cs - gCs/gamma;
			
			// compute objective at line search point
			Fzp = 0.0;
			for (task=0; task<n_tasks; task++)
			{
				for (int i=options.ind[task]; i<options.ind[task+1]; i++)
				{
					double aa = -y[i]*(features->dense_dot(i, Wzp.col(task).data(), n_feats)+Cs[task]);
					double bb = CMath::max(aa,0.0);

					Fzp += (CMath::log(CMath::exp(-bb) + CMath::exp(aa-bb)) + bb)/n_vecs;
				}
			}
			Fzp += Wzp.squaredNorm();

			// compute delta between line search point and search point
			delta_Wzp = Wzp - Ws;
			delta_Czp = Czp - Cs;

			// norms of delta
			double nrm_delta_Wzp = delta_Wzp.squaredNorm();
			double nrm_delta_Czp = delta_Czp.squaredNorm();

			double r_sum = (nrm_delta_Wzp + nrm_delta_Czp)/2;

			double Fzp_gamma = Fs + (delta_Wzp.transpose()*gWs).trace() +
				(delta_Czp.transpose()*gCs).trace() +
				(gamma/2)*nrm_delta_Wzp +
				(gamma/2)*nrm_delta_Czp;

			// break if delta is getting too small
			if (r_sum <= 1e-20)
			{
				gradient_break = true;
				break;
			}

			// break if objective at line searc point is smaller than Fzp_gamma
			if (Fzp <= Fzp_gamma)
				break;
			else
				gamma *= gamma_inc;
		}

		if (gradient_break)
			break;

		Wz_old = Wz;
		Cz_old = Cz;
		Wz = Wzp;
		Cz = Czp;

		// compute objective value
		obj = Fzp;
		for (task=0; task<n_tasks; task++)
			obj += rho1*(Wz.col(task).norm());

		// check if process should be terminated 
		switch (options.termination)
		{
			case 0:
				if (iter>=2)
				{
					if ( (CMath::abs(obj)-CMath::abs(obj_old)) <= options.tolerance)
						break;
				}
			break;
			case 1:
				if (iter>=2)
				{
					if ( (CMath::abs(obj)-CMath::abs(obj_old)) <= options.tolerance*CMath::abs(obj_old))
						break;
				}
			break;
			case 2:
				if (CMath::abs(obj) <= options.tolerance)
					break;
			break;
			case 3:
				if (iter>=options.max_iter)
					break;
			break;
		}

		iter++;
		t_old = t;
		t = 0.5 * (1 + CMath::sqrt(1.0 + 4*t*t));
	}
	SG_SDEBUG("%d iteration passed\n",iter);

	SGMatrix<float64_t> tasks_w(n_feats, n_tasks);
	for (int i=0; i<n_feats; i++)
	{
		for (int task=0; task<n_tasks; task++)
			tasks_w[i] = Wzp(i,task);
	}
	SGVector<float64_t> tasks_c(n_tasks);
	for (int i=0; i<n_tasks; i++) tasks_c[i] = Czp[i];
	return slep_result_t(tasks_w, tasks_c);
};
};
