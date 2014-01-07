
/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Copyright (C) 2012 Soeren Sonnenburg
 */

#include <lib/config.h>
#ifdef HAVE_LAPACK
#include <regression/svr/LibLinearRegression.h>
#include <mathematics/Math.h>
#include <labels/RegressionLabels.h>
#include <optimization/liblinear/tron.h>
#include <lib/Signal.h>

using namespace shogun;

CLibLinearRegression::CLibLinearRegression() :
	CLinearMachine()
{
	init_defaults();
}

CLibLinearRegression::CLibLinearRegression(float64_t C, CDotFeatures* feats, CLabels* labs) :
	CLinearMachine()
{
	init_defaults();
	set_C(C);
	set_features(feats);
	set_labels(labs);
}

void CLibLinearRegression::init_defaults()
{
	set_C(1.0);
	set_epsilon(1e-2);
	set_max_iter(10000);
	set_use_bias(false);
	set_liblinear_regression_type(L2R_L1LOSS_SVR_DUAL);
}

void CLibLinearRegression::register_parameters()
{
	SG_ADD(&m_C, "m_C", "regularization constant",MS_AVAILABLE);
	SG_ADD(&m_epsilon, "m_epsilon", "tolerance epsilon",MS_NOT_AVAILABLE);
	SG_ADD(&m_epsilon, "m_tube_epsilon", "svr tube epsilon",MS_AVAILABLE);
	SG_ADD(&m_max_iter, "m_max_iter", "max number of iterations",MS_NOT_AVAILABLE);
	SG_ADD(&m_use_bias, "m_use_bias", "indicates whether bias should be used",MS_NOT_AVAILABLE);
}

CLibLinearRegression::~CLibLinearRegression()
{
}

bool CLibLinearRegression::train_machine(CFeatures* data)
{
	CSignal::clear_cancel();

	if (data)
		set_features((CDotFeatures*)data);

	ASSERT(features)
	ASSERT(m_labels && m_labels->get_label_type()==LT_REGRESSION)

	int32_t num_train_labels=m_labels->get_num_labels();
	int32_t num_feat=features->get_dim_feature_space();
	int32_t num_vec=features->get_num_vectors();

	if (num_vec!=num_train_labels)
	{
		SG_ERROR("number of vectors %d does not match "
				"number of training labels %d\n",
				num_vec, num_train_labels);
	}

	if (m_use_bias)
		w=SGVector<float64_t>(SG_MALLOC(float64_t, num_feat+1), num_feat);
	else
		w=SGVector<float64_t>(num_feat);

	liblinear_problem prob;
	if (m_use_bias)
	{
		prob.n=w.vlen+1;
		memset(w.vector, 0, sizeof(float64_t)*(w.vlen+1));
	}
	else
	{
		prob.n=w.vlen;
		memset(w.vector, 0, sizeof(float64_t)*(w.vlen+0));
	}
	prob.l=num_vec;
	prob.x=features;
	prob.y=SG_MALLOC(float64_t, prob.l);

	switch (m_liblinear_regression_type)
	{
		case L2R_L2LOSS_SVR:
		{
			double* Cs = SG_MALLOC(double, prob.l);
			for(int i = 0; i < prob.l; i++)
				Cs[i] = m_C;

			function *fun_obj=new l2r_l2_svr_fun(&prob, Cs, m_tube_epsilon);
			CTron tron_obj(fun_obj, m_epsilon);
			tron_obj.tron(w.vector, m_max_train_time);
			delete fun_obj;
			SG_FREE(Cs);
			break;

		}
		case L2R_L1LOSS_SVR_DUAL:
			solve_l2r_l1l2_svr(&prob);
			break;
		case L2R_L2LOSS_SVR_DUAL:
			solve_l2r_l1l2_svr(&prob);
			break;
		default:
			SG_ERROR("Error: unknown regression type\n")
			break;
	}

	return true;
}

// A coordinate descent algorithm for
// L1-loss and L2-loss epsilon-SVR dual problem
//
//  min_\beta  0.5\beta^T (Q + diag(lambda)) \beta - p \sum_{i=1}^l|\beta_i| + \sum_{i=1}^l yi\beta_i,
//    s.t.      -upper_bound_i <= \beta_i <= upper_bound_i,
//
//  where Qij = xi^T xj and
//  D is a diagonal matrix
//
// In L1-SVM case:
//		upper_bound_i = C
//		lambda_i = 0
// In L2-SVM case:
//		upper_bound_i = INF
//		lambda_i = 1/(2*C)
//
// Given:
// x, y, p, C
// eps is the stopping tolerance
//
// solution will be put in w
//
// See Algorithm 4 of Ho and Lin, 2012

#undef GETI
#define GETI(i) (0)
// To support weights for instances, use GETI(i) (i)

void CLibLinearRegression::solve_l2r_l1l2_svr(const liblinear_problem *prob)
{
	int l = prob->l;
	double C = m_C;
	double p = m_tube_epsilon;
	int w_size = prob->n;
	double eps = m_epsilon;
	int i, s, iter = 0;
	int max_iter = 1000;
	int active_size = l;
	int *index = new int[l];

	double d, G, H;
	double Gmax_old = CMath::INFTY;
	double Gmax_new, Gnorm1_new;
	double Gnorm1_init = 0.0;
	double *beta = new double[l];
	double *QD = new double[l];
	double *y = prob->y;

	// L2R_L2LOSS_SVR_DUAL
	double lambda[1], upper_bound[1];
	lambda[0] = 0.5/C;
	upper_bound[0] = CMath::INFTY;

	if(m_liblinear_regression_type == L2R_L1LOSS_SVR_DUAL)
	{
		lambda[0] = 0;
		upper_bound[0] = C;
	}

	// Initial beta can be set here. Note that
	// -upper_bound <= beta[i] <= upper_bound
	for(i=0; i<l; i++)
		beta[i] = 0;

	for(i=0; i<w_size; i++)
		w[i] = 0;
	for(i=0; i<l; i++)
	{
		QD[i] = prob->x->dot(i, prob->x,i);
		prob->x->add_to_dense_vec(beta[i], i, w.vector, w_size);

		if (prob->use_bias)
			w.vector[w_size]+=beta[i];

		index[i] = i;
	}


	while(iter < max_iter)
	{
		Gmax_new = 0;
		Gnorm1_new = 0;

		for(i=0; i<active_size; i++)
		{
			int j = CMath::random(i, active_size-1);
			CMath::swap(index[i], index[j]);
		}

		for(s=0; s<active_size; s++)
		{
			i = index[s];
			G = -y[i] + lambda[GETI(i)]*beta[i];
			H = QD[i] + lambda[GETI(i)];

			G += prob->x->dense_dot(i, w.vector, w_size);
			if (prob->use_bias)
				G+=w.vector[w_size];

			double Gp = G+p;
			double Gn = G-p;
			double violation = 0;
			if(beta[i] == 0)
			{
				if(Gp < 0)
					violation = -Gp;
				else if(Gn > 0)
					violation = Gn;
				else if(Gp>Gmax_old && Gn<-Gmax_old)
				{
					active_size--;
					CMath::swap(index[s], index[active_size]);
					s--;
					continue;
				}
			}
			else if(beta[i] >= upper_bound[GETI(i)])
			{
				if(Gp > 0)
					violation = Gp;
				else if(Gp < -Gmax_old)
				{
					active_size--;
					CMath::swap(index[s], index[active_size]);
					s--;
					continue;
				}
			}
			else if(beta[i] <= -upper_bound[GETI(i)])
			{
				if(Gn < 0)
					violation = -Gn;
				else if(Gn > Gmax_old)
				{
					active_size--;
					CMath::swap(index[s], index[active_size]);
					s--;
					continue;
				}
			}
			else if(beta[i] > 0)
				violation = fabs(Gp);
			else
				violation = fabs(Gn);

			Gmax_new = CMath::max(Gmax_new, violation);
			Gnorm1_new += violation;

			// obtain Newton direction d
			if(Gp < H*beta[i])
				d = -Gp/H;
			else if(Gn > H*beta[i])
				d = -Gn/H;
			else
				d = -beta[i];

			if(fabs(d) < 1.0e-12)
				continue;

			double beta_old = beta[i];
			beta[i] = CMath::min(CMath::max(beta[i]+d, -upper_bound[GETI(i)]), upper_bound[GETI(i)]);
			d = beta[i]-beta_old;

			if(d != 0)
			{
				prob->x->add_to_dense_vec(d, i, w.vector, w_size);

				if (prob->use_bias)
					w.vector[w_size]+=d;
			}
		}

		if(iter == 0)
			Gnorm1_init = Gnorm1_new;
		iter++;

		SG_SABS_PROGRESS(Gnorm1_new, -CMath::log10(Gnorm1_new), -CMath::log10(eps*Gnorm1_init), -CMath::log10(Gnorm1_init), 6)

		if(Gnorm1_new <= eps*Gnorm1_init)
		{
			if(active_size == l)
				break;
			else
			{
				active_size = l;
				Gmax_old = CMath::INFTY;
				continue;
			}
		}

		Gmax_old = Gmax_new;
	}

	SG_DONE()
	SG_INFO("\noptimization finished, #iter = %d\n", iter)
	if(iter >= max_iter)
		SG_INFO("\nWARNING: reaching max number of iterations\nUsing -s 11 may be faster\n\n")

	// calculate objective value
	double v = 0;
	int nSV = 0;
	for(i=0; i<w_size; i++)
		v += w[i]*w[i];
	v = 0.5*v;
	for(i=0; i<l; i++)
	{
		v += p*fabs(beta[i]) - y[i]*beta[i] + 0.5*lambda[GETI(i)]*beta[i]*beta[i];
		if(beta[i] != 0)
			nSV++;
	}

	SG_INFO("Objective value = %lf\n", v)
	SG_INFO("nSV = %d\n",nSV)

	delete [] beta;
	delete [] QD;
	delete [] index;
}

#endif /* HAVE_LAPACK */
