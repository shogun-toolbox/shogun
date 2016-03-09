/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2012 Sergey Lisitsyn
 * Copyright (C) 2012 Jiayu Zhou and Jieping Ye
 */

#include <shogun/lib/malsar/malsar_joint_feature_learning.h>
#include <shogun/lib/Signal.h>
#include <shogun/mathematics/Math.h>
#include <shogun/mathematics/eigen3.h>
#include <iostream>

using namespace Eigen;
using namespace std;

namespace shogun
{

malsar_result_t malsar_joint_feature_learning(
		CDotFeatures* features,
		double* y,
		double rho1,
		double rho2,
		const malsar_options& options)
{
	int task;
	int n_feats = features->get_dim_feature_space();
	SG_SDEBUG("n feats = %d\n", n_feats)
	int n_vecs = features->get_num_vectors();
	SG_SDEBUG("n vecs = %d\n", n_vecs)
	int n_tasks = options.n_tasks;
	SG_SDEBUG("n tasks = %d\n", n_tasks)

	int iter = 0;

	// initialize weight vector and bias for each task
	MatrixXd Ws = MatrixXd::Zero(n_feats, n_tasks);
	VectorXd Cs = VectorXd::Zero(n_tasks);
	MatrixXd Wz=Ws, Wzp=Ws, Wz_old=Ws, delta_Wzp=Ws, gWs=Ws;
	VectorXd Cz=Cs, Czp=Cs, Cz_old=Cs, delta_Czp=Cs, gCs=Cs;

	double t=1, t_old=0;
	double gamma=1, gamma_inc=2;
	double obj=0.0, obj_old=0.0;

	//internal::set_is_malloc_allowed(false);
	bool done = false;
	while (!done && iter <= options.max_iter && !CSignal::cancel_computations())
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
			SGVector<index_t> task_idx = options.tasks_indices[task];
			int n_task_vecs = task_idx.vlen;
			for (int i=0; i<n_task_vecs; i++)
			{
				double aa = -y[task_idx[i]]*(features->dense_dot(task_idx[i], Ws.col(task).data(), n_feats)+Cs[task]);
				double bb = CMath::max(aa,0.0);

				// avoid underflow when computing exponential loss
				Fs += (CMath::log(CMath::exp(-bb) + CMath::exp(aa-bb)) + bb)/n_task_vecs;
				double b = -y[task_idx[i]]*(1 - 1/(1+CMath::exp(aa)))/n_task_vecs;

				gCs[task] += b;
				features->add_to_dense_vec(b, task_idx[i], gWs.col(task).data(), n_feats);
			}
		}
		gWs.noalias() += 2*rho2*Ws;

		// add regularizer
		Fs += Ws.squaredNorm();

		//cout << "gWs" << endl << gWs << endl;
		//cout << "gCs" << endl << gCs << endl;
		//SG_SPRINT("Fs = %f\n",Fs)

		double Fzp = 0.0;

		int inner_iter = 0;
		// line search, Armijo-Goldstein scheme
		while (inner_iter <= 1000)
		{
			// compute lasso projection of Ws - gWs/gamma
			for (int i=0; i<n_feats; i++)
			{
				Wzp.row(i).noalias() = Ws.row(i) - gWs.row(i)/gamma;
				double norm = Wzp.row(i).lpNorm<2>();
				if (norm == 0.0)
					Wzp.row(i).setZero();
				else
				{
					double threshold = norm - rho1/gamma;
					if (threshold < 0.0)
						Wzp.row(i).setZero();
					else
						Wzp.row(i) *= threshold/norm;
				}
			}
			// walk in direction of antigradient
			Czp = Cs - gCs/gamma;

			// compute objective at line search point
			Fzp = 0.0;
			for (task=0; task<n_tasks; task++)
			{
				SGVector<index_t> task_idx = options.tasks_indices[task];
				int n_task_vecs = task_idx.vlen;
				for (int i=0; i<n_task_vecs; i++)
				{
					double aa = -y[task_idx[i]]*(features->dense_dot(task_idx[i], Wzp.col(task).data(), n_feats)+Czp[task]);
					double bb = CMath::max(aa,0.0);

					Fzp += (CMath::log(CMath::exp(-bb) + CMath::exp(aa-bb)) + bb)/n_task_vecs;
				}
			}
			Fzp += rho2*Wzp.squaredNorm();

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
				SG_SDEBUG("Line search point is too close to search point\n")
				done = true;
				break;
			}

			// break if objective at line searc point is smaller than Fzp_gamma
			if (Fzp <= Fzp_gamma)
				break;
			else
				gamma *= gamma_inc;

			inner_iter++;
		}

		Wz_old = Wz;
		Cz_old = Cz;
		Wz = Wzp;
		Cz = Czp;

		// compute objective value
		obj_old = obj;
		obj = Fzp;
		for (int i=0; i<n_feats; i++)
			obj += rho1*(Wz.row(i).lpNorm<2>());
		//for (task=0; task<n_tasks; task++)
		//	obj += rho1*(Wz.col(task).norm());
		SG_SDEBUG("Obj = %f\n",obj)
		//SG_SABS_PROGRESS(obj,0.0)
		// check if process should be terminated
		switch (options.termination)
		{
			case 0:
				if (iter>=2)
				{
					if ( CMath::abs(obj-obj_old) <= options.tolerance )
					{
						SG_SDEBUG("Objective changes less than tolerance\n")
						done = true;
					}
				}
			break;
			case 1:
				if (iter>=2)
				{
					if ( CMath::abs(obj-obj_old) <= options.tolerance*CMath::abs(obj_old))
						done = true;
				}
			break;
			case 2:
				if (CMath::abs(obj) <= options.tolerance)
					done = true;
			break;
			case 3:
				if (iter>=options.max_iter)
					done = true;
			break;
		}

		iter++;
		t_old = t;
		t = 0.5 * (1 + CMath::sqrt(1.0 + 4*t*t));
	}
	//internal::set_is_malloc_allowed(true);
	SG_SDONE()
	SG_SDEBUG("%d iteration passed, objective = %f\n",iter,obj)

	SGMatrix<float64_t> tasks_w(n_feats, n_tasks);
	for (int i=0; i<n_feats; i++)
	{
		for (task=0; task<n_tasks; task++)
			tasks_w(i,task) = Wzp(i,task);
	}
	SGVector<float64_t> tasks_c(n_tasks);
	for (int i=0; i<n_tasks; i++) tasks_c[i] = Czp[i];
	return malsar_result_t(tasks_w, tasks_c);
};
};
