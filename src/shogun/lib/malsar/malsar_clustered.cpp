/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2012 Sergey Lisitsyn
 * Copyright (C) 2012 Jiayu Zhou and Jieping Ye
 */

#include <shogun/lib/malsar/malsar_clustered.h>
#ifdef HAVE_EIGEN3
#ifndef HAVE_CXX11
#include <shogun/mathematics/Math.h>
#include <shogun/mathematics/eigen3.h>
#include <iostream>
#include <shogun/lib/external/libqp.h>

using namespace Eigen;
using namespace std;

namespace shogun
{

static double* H_diag_matrix;
static int H_diag_matrix_ld;

static const double* get_col(uint32_t j)
{
	return H_diag_matrix + j*H_diag_matrix_ld;
}

malsar_result_t malsar_clustered(
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

	H_diag_matrix = SG_CALLOC(double, n_tasks*n_tasks);
	for (int i=0; i<n_tasks; i++)
		H_diag_matrix[i*n_tasks+i] = 2.0;
	H_diag_matrix_ld = n_tasks;

	int iter = 0;

	// initialize weight vector and bias for each task
	MatrixXd Ws = MatrixXd::Zero(n_feats, n_tasks);
	VectorXd Cs = VectorXd::Zero(n_tasks);
	MatrixXd Ms = MatrixXd::Identity(n_tasks, n_tasks)*options.n_clusters/n_tasks;
	MatrixXd IM = Ms;
	MatrixXd IMsqinv = Ms;
	MatrixXd invEtaMWt = Ms;

	MatrixXd Wz=Ws, Wzp=Ws, Wz_old=Ws, delta_Wzp=Ws, gWs=Ws;
	VectorXd Cz=Cs, Czp=Cs, Cz_old=Cs, delta_Czp=Cs, gCs=Cs;
	MatrixXd Mz=Ms, Mzp=Ms, Mz_old=Ms, delta_Mzp=Ms, gMs=Ms;
	MatrixXd Mzp_Pz;

	double eta = rho2/rho1;
	double c = rho1*eta*(1+eta);

	double t=1, t_old=0;
	double gamma=1, gamma_inc=2;
	double obj=0.0, obj_old=0.0;

	double* diag_H = SG_MALLOC(double, n_tasks);
	double* f = SG_MALLOC(double, n_tasks);
	double* a = SG_MALLOC(double, n_tasks);
	double* lb = SG_MALLOC(double, n_tasks);
	double* ub = SG_MALLOC(double, n_tasks);
	double* x = SG_CALLOC(double, n_tasks);

	//internal::set_is_malloc_allowed(false);
	bool done = false;
	while (!done && iter <= options.max_iter)
	{
		double alpha = double(t_old - 1)/t;
		SG_SDEBUG("alpha=%f\n",alpha)

		// compute search point
		Ws = (1+alpha)*Wz - alpha*Wz_old;
		Cs = (1+alpha)*Cz - alpha*Cz_old;
		Ms = (1+alpha)*Mz - alpha*Mz_old;

		// zero gradient
		gWs.setZero();
		gCs.setZero();
		//internal::set_is_malloc_allowed(true);
		SG_SDEBUG("Computing gradient\n")
		IM = (eta*MatrixXd::Identity(n_tasks,n_tasks)+Ms);
		IMsqinv = (IM*IM).inverse();
		invEtaMWt = IM.inverse()*Ws.transpose();
		gMs.noalias() = -c*(Ws.transpose()*Ws)*IMsqinv;
		gWs.noalias() += 2*c*invEtaMWt.transpose();
		//internal::set_is_malloc_allowed(false);

		// compute gradient and objective at search point
		double Fs = 0;
		for (task=0; task<n_tasks; task++)
		{
			SGVector<index_t> task_idx = options.tasks_indices[task];
			int n_vecs_task = task_idx.vlen;
			for (int i=0; i<n_vecs_task; i++)
			{
				double aa = -y[task_idx[i]]*(features->dense_dot(task_idx[i], Ws.col(task).data(), n_feats)+Cs[task]);
				double bb = CMath::max(aa,0.0);

				// avoid underflow when computing exponential loss
				Fs += (CMath::log(CMath::exp(-bb) + CMath::exp(aa-bb)) + bb)/n_vecs_task;
				double b = -y[task_idx[i]]*(1 - 1/(1+CMath::exp(aa)))/n_vecs_task;
				gCs[task] += b;
				features->add_to_dense_vec(b, task_idx[i], gWs.col(task).data(), n_feats);
			}
		}
		SG_SDEBUG("Computed gradient\n")

		// add regularizer
		Fs += c*(Ws*invEtaMWt).trace();
		SG_SDEBUG("Fs = %f \n", Fs)

		double Fzp = 0.0;

		int inner_iter = 0;
		// line search, Armijo-Goldstein scheme
		while (inner_iter <= 1)
		{
			Wzp = Ws - gWs/gamma;
			Czp = Cs - gCs/gamma;
			// compute singular projection of Ms - gMs/gamma with k
			//internal::set_is_malloc_allowed(true);
			EigenSolver<MatrixXd> eigensolver;
			eigensolver.compute(Ms-gMs/gamma, true);
			if (eigensolver.info()!=Eigen::Success)
				SG_SERROR("Eigendecomposition failed")

			// solve problem
			// min sum_i (s_i - s*_i)^2 s.t. sum_i s_i = k, 0<=s_i<=1
			for (int i=0; i<n_tasks; i++)
			{
				diag_H[i] = 2.0;
				// TODO fails with C++11
				//std::complex<MatrixXd::Scalar> eigenvalue = eigensolver.eigenvalues()[i];
				//cout << "eigenvalue " << eigenvalue << "=" << std::real(eigenvalue) << "+i" << std::imag(eigenvalue) << endl;
				f[i] = -2*eigensolver.eigenvalues()[i].real();
				if (f[i]!=f[i])
					SG_SERROR("NaN %d eigenvalue", i)

				SG_SDEBUG("%dth eigenvalue %f\n",i,eigensolver.eigenvalues()[i].real())
				a[i] = 1.0;
				lb[i] = 0.0;
				ub[i] = 1.0;
				x[i] = double(options.n_clusters)/n_tasks;
			}
			double b = options.n_clusters;//eigensolver.eigenvalues().sum().real();
			SG_SDEBUG("b = %f\n", b)
			SG_SDEBUG("Calling libqp\n")
			libqp_state_T problem_state = libqp_gsmo_solver(&get_col,diag_H,f,a,b,lb,ub,x,n_tasks,1000,1e-6,NULL);
			SG_SDEBUG("Libqp objective = %f\n",problem_state.QP)
			SG_SDEBUG("Exit code = %d\n",problem_state.exitflag)
			SG_SDEBUG("%d iteration passed\n",problem_state.nIter)
			SG_SDEBUG("Solution is \n [ ")
			for (int i=0; i<n_tasks; i++)
				SG_SDEBUG("%f ", x[i])
			SG_SDEBUG("]\n")
			Map<VectorXd> Mzp_DiagSigz(x,n_tasks);
			Mzp_Pz = eigensolver.eigenvectors().real();
			Mzp = Mzp_Pz*Mzp_DiagSigz.asDiagonal()*Mzp_Pz.transpose();
			//internal::set_is_malloc_allowed(false);
			// walk in direction of antigradient
			for (int i=0; i<n_tasks; i++)
				Mzp_DiagSigz[i] += eta;
			//internal::set_is_malloc_allowed(true);
			invEtaMWt = (Mzp_Pz*
			             (Mzp_DiagSigz.cwiseInverse().asDiagonal())*
			             Mzp_Pz.transpose())*
			             Wzp.transpose();
			//internal::set_is_malloc_allowed(false);
			// compute objective at line search point
			Fzp = 0.0;
			for (task=0; task<n_tasks; task++)
			{
				SGVector<index_t> task_idx = options.tasks_indices[task];
				int n_vecs_task = task_idx.vlen;
				for (int i=0; i<n_vecs_task; i++)
				{
					double aa = -y[task_idx[i]]*(features->dense_dot(task_idx[i], Wzp.col(task).data(), n_feats)+Cs[task]);
					double bb = CMath::max(aa,0.0);

					Fzp += (CMath::log(CMath::exp(-bb) + CMath::exp(aa-bb)) + bb)/n_vecs_task;
				}
			}
			Fzp += c*(Wzp*invEtaMWt).trace();

			// compute delta between line search point and search point
			delta_Wzp = Wzp - Ws;
			delta_Czp = Czp - Cs;
			delta_Mzp = Mzp - Ms;

			// norms of delta
			double nrm_delta_Wzp = delta_Wzp.squaredNorm();
			double nrm_delta_Czp = delta_Czp.squaredNorm();
			double nrm_delta_Mzp = delta_Mzp.squaredNorm();

			double r_sum = (nrm_delta_Wzp + nrm_delta_Czp + nrm_delta_Mzp)/3;

			double Fzp_gamma = 0.0;
			if (n_feats > n_tasks)
			{
				Fzp_gamma = Fs + (delta_Wzp.transpose()*gWs).trace() +
				                 (delta_Czp.transpose()*gCs).trace() +
				                 (delta_Mzp.transpose()*gMs).trace() +
				                 (gamma/2)*nrm_delta_Wzp +
				                 (gamma/2)*nrm_delta_Czp +
				                 (gamma/2)*nrm_delta_Mzp;
			}
			else
			{
				Fzp_gamma = Fs + (gWs.transpose()*delta_Wzp).trace() +
				                 (gCs.transpose()*delta_Czp).trace() +
				                 (gMs.transpose()*delta_Mzp).trace() +
				                 (gamma/2)*nrm_delta_Wzp +
				                 (gamma/2)*nrm_delta_Czp +
				                 (gamma/2)*nrm_delta_Mzp;
			}

			// break if delta is getting too small
			if (r_sum <= 1e-20)
			{
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
		Mz_old = Mz;
		Wz = Wzp;
		Cz = Czp;
		Mz = Mzp;

		// compute objective value
		obj_old = obj;
		obj = Fzp;

		// check if process should be terminated
		switch (options.termination)
		{
			case 0:
				if (iter>=2)
				{
					if ( CMath::abs(obj-obj_old) <= options.tolerance )
						done = true;
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
	SG_SDEBUG("%d iteration passed, objective = %f\n",iter,obj)

	SG_FREE(H_diag_matrix);
	SG_FREE(diag_H);
	SG_FREE(f);
	SG_FREE(a);
	SG_FREE(lb);
	SG_FREE(ub);
	SG_FREE(x);

	SGMatrix<float64_t> tasks_w(n_feats, n_tasks);
	for (int i=0; i<n_feats; i++)
	{
		for (task=0; task<n_tasks; task++)
			tasks_w(i,task) = Wzp(i,task);
	}
	//tasks_w.display_matrix();
	SGVector<float64_t> tasks_c(n_tasks);
	for (int i=0; i<n_tasks; i++) tasks_c[i] = Czp[i];
	return malsar_result_t(tasks_w, tasks_c);
};
};
#endif
#endif
