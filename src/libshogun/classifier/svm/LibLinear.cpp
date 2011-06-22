/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2007-2010 Soeren Sonnenburg
 * Copyright (c) 2007-2009 The LIBLINEAR Project.
 * Copyright (C) 2007-2010 Fraunhofer Institute FIRST and Max-Planck-Society
 */
#include "lib/config.h"

#ifdef HAVE_LAPACK
#include "lib/io.h"
#include "lib/Signal.h"
#include "lib/Time.h"
#include "base/Parameter.h"
#include "classifier/svm/LibLinear.h"
#include "classifier/svm/SVM_linear.h"
#include "classifier/svm/Tron.h"
#include "features/DotFeatures.h"

using namespace shogun;

CLibLinear::CLibLinear(void)
: CLinearMachine()
{
    init();
}

CLibLinear::CLibLinear(LIBLINEAR_SOLVER_TYPE l)
: CLinearMachine()
{
    init();
}

CLibLinear::CLibLinear(
	float64_t C, CDotFeatures* traindat, CLabels* trainlab)
: CLinearMachine()
{
    init();
	C1=C;
	C2=C;
	use_bias=true;

	set_features(traindat);
	set_labels(trainlab);
	init_linear_term();
}

void CLibLinear::init()
{
	liblinear_solver_type=L2R_L1LOSS_SVC_DUAL;
	use_bias=false;
	C1=1;
	C2=1;
	set_max_iterations();
	m_linear_term=NULL;
	m_linear_term_len=0;
	epsilon=1e-5;

    m_parameters->add(&C1, "C1",  "C Cost constant 1.");
    m_parameters->add(&C2, "C2",  "C Cost constant 2.");
    m_parameters->add(&use_bias, "use_bias",  "Indicates if bias is used.");
    m_parameters->add(&epsilon, "epsilon",  "Convergence precision.");
    m_parameters->add(&max_iterations, "max_iterations",  "Max number of iterations.");
    m_parameters->add_vector(&m_linear_term, &m_linear_term_len, "linear_term", "Linear Term");
    m_parameters->add((machine_int_t*) &liblinear_solver_type, "liblinear_solver_type", "Type of LibLinear solver.");
}

CLibLinear::~CLibLinear()
{
	delete[] m_linear_term;
}

bool CLibLinear::train(CFeatures* data)
{
	CSignal::clear_cancel();
	ASSERT(labels);

	if (data)
	{
		if (!data->has_property(FP_DOT))
			SG_ERROR("Specified features are not of type CDotFeatures\n");

		set_features((CDotFeatures*) data);
	}
	ASSERT(features);
	ASSERT(labels->is_two_class_labeling());


	int32_t num_train_labels=labels->get_num_labels();
	int32_t num_feat=features->get_dim_feature_space();
	int32_t num_vec=features->get_num_vectors();

	if (liblinear_solver_type == L1R_L2LOSS_SVC ||
			(liblinear_solver_type == L1R_LR) )
	{
		if (num_feat!=num_train_labels)
		{
			SG_ERROR("L1 methods require the data to be transposed: "
					"number of features %d does not match number of "
					"training labels %d\n",
					num_feat, num_train_labels);
		}
		CMath::swap(num_feat, num_vec);

	}
	else
	{
		if (num_vec!=num_train_labels)
		{
			SG_ERROR("number of vectors %d does not match "
					"number of training labels %d\n",
					num_vec, num_train_labels);
		}
	}
	delete[] w;
	if (use_bias)
		w=new float64_t[num_feat+1];
	else
		w=new float64_t[num_feat+0];
	w_dim=num_feat;

	problem prob;
	if (use_bias)
	{
		prob.n=w_dim+1;
		memset(w, 0, sizeof(float64_t)*(w_dim+1));
	}
	else
	{
		prob.n=w_dim;
		memset(w, 0, sizeof(float64_t)*(w_dim+0));
	}
	prob.l=num_vec;
	prob.x=features;
	prob.y=new int[prob.l];
	prob.use_bias=use_bias;

	for (int32_t i=0; i<prob.l; i++)
		prob.y[i]=labels->get_int_label(i);

	int pos = 0;
	int neg = 0;
	for(int i=0;i<prob.l;i++)
	{
		if(prob.y[i]==+1)
			pos++;
	}
	neg = prob.l - pos;

	SG_INFO("%d training points %d dims\n", prob.l, prob.n);

	function *fun_obj=NULL;
	double Cp=C1;
	double Cn=C2;
	switch (liblinear_solver_type)
	{
		case L2R_LR:
		{
			fun_obj=new l2r_lr_fun(&prob, Cp, Cn);
			CTron tron_obj(fun_obj, epsilon*CMath::min(pos,neg)/prob.l, max_iterations);
			SG_DEBUG("starting L2R_LR training via tron\n");
			tron_obj.tron(w, max_train_time);
			SG_DEBUG("done with tron\n");
			delete fun_obj;
			break;
		}
		case L2R_L2LOSS_SVC:
		{
			fun_obj=new l2r_l2_svc_fun(&prob, Cp, Cn);
			CTron tron_obj(fun_obj, epsilon*CMath::min(pos,neg)/prob.l, max_iterations);
			tron_obj.tron(w, max_train_time);
			delete fun_obj;
			break;
		}
		case L2R_L2LOSS_SVC_DUAL:
			solve_l2r_l1l2_svc(&prob, epsilon, Cp, Cn, L2R_L2LOSS_SVC_DUAL);
			break;
		case L2R_L1LOSS_SVC_DUAL:
			solve_l2r_l1l2_svc(&prob, epsilon, Cp, Cn, L2R_L1LOSS_SVC_DUAL);
			break;
		case L1R_L2LOSS_SVC:
		{
			//ASSUME FEATURES ARE TRANSPOSED ALREADY
			solve_l1r_l2_svc(&prob, epsilon*CMath::min(pos,neg)/prob.l, Cp, Cn);
			break;
		}
		case L1R_LR:
		{
			//ASSUME FEATURES ARE TRANSPOSED ALREADY
			solve_l1r_lr(&prob, epsilon*CMath::min(pos,neg)/prob.l, Cp, Cn);
			break;
		}
		case MCSVM_CS:
		{
			SG_NOTIMPLEMENTED;
			/* TODO...
			model_->w=Malloc(double, n*nr_class);
			for(i=0;i<nr_class;i++)
				for(j=start[i];j<start[i]+count[i];j++)
					sub_prob.y[j] = i;
			Solver_MCSVM_CS Solver(&sub_prob, nr_class, weighted_C, param->eps);
			Solver.Solve(model_->w);
			*/
		}
		default:
			SG_ERROR("Error: unknown solver_type\n");
			break;
	}

	if (use_bias)
		set_bias(w[w_dim]);
	else
		set_bias(0);

    delete[] prob.y;

	return true;
}

// A coordinate descent algorithm for
// L1-loss and L2-loss SVM dual problems
//
//  min_\alpha  0.5(\alpha^T (Q + D)\alpha) - e^T \alpha,
//    s.t.      0 <= alpha_i <= upper_bound_i,
//
//  where Qij = yi yj xi^T xj and
//  D is a diagonal matrix
//
// In L1-SVM case:
// 		upper_bound_i = Cp if y_i = 1
// 		upper_bound_i = Cn if y_i = -1
// 		D_ii = 0
// In L2-SVM case:
// 		upper_bound_i = INF
// 		D_ii = 1/(2*Cp)	if y_i = 1
// 		D_ii = 1/(2*Cn)	if y_i = -1
//
// Given:
// x, y, Cp, Cn
// eps is the stopping tolerance
//
// solution will be put in w

#undef GETI
#define GETI(i) (y[i]+1)
// To support weights for instances, use GETI(i) (i)

void CLibLinear::solve_l2r_l1l2_svc(
			const problem *prob, double eps, double Cp, double Cn, LIBLINEAR_SOLVER_TYPE st)
{
	int l = prob->l;
	int w_size = prob->n;
	int i, s, iter = 0;
	double C, d, G;
	double *QD = new double[l];
	int *index = new int[l];
	double *alpha = new double[l];
	int32_t *y = new int32_t[l];
	int active_size = l;

	// PG: projected gradient, for shrinking and stopping
	double PG;
	double PGmax_old = CMath::INFTY;
	double PGmin_old = -CMath::INFTY;
	double PGmax_new, PGmin_new;

	// default solver_type: L2R_L2LOSS_SVC_DUAL
	double diag[3] = {0.5/Cn, 0, 0.5/Cp};
	double upper_bound[3] = {CMath::INFTY, 0, CMath::INFTY};
	if(st == L2R_L1LOSS_SVC_DUAL)
	{
		diag[0] = 0;
		diag[2] = 0;
		upper_bound[0] = Cn;
		upper_bound[2] = Cp;
	}

	int n = prob->n;

	if (prob->use_bias)
		n--;

	for(i=0; i<w_size; i++)
		w[i] = 0;

	for(i=0; i<l; i++)
	{
		alpha[i] = 0;
		if(prob->y[i] > 0)
		{
			y[i] = +1;
		}
		else
		{
			y[i] = -1;
		}
		QD[i] = diag[GETI(i)];

		QD[i] += prob->x->dot(i, prob->x,i);
		index[i] = i;
	}


	CTime start_time;
	while (iter < max_iterations && !CSignal::cancel_computations())
	{
		if (max_train_time > 0 && start_time.cur_time_diff() > max_train_time)
		  break;

		PGmax_new = -CMath::INFTY;
		PGmin_new = CMath::INFTY;

		for (i=0; i<active_size; i++)
		{
			int j = i+rand()%(active_size-i);
			CMath::swap(index[i], index[j]);
		}

		for (s=0;s<active_size;s++)
		{
			i = index[s];
			int32_t yi = y[i];

			G = prob->x->dense_dot(i, w, n);
			if (prob->use_bias)
				G+=w[n];

			if (m_linear_term)
				G = G*yi + m_linear_term[i];
			else
				G = G*yi-1;

			C = upper_bound[GETI(i)];
			G += alpha[i]*diag[GETI(i)];

			PG = 0;
			if (alpha[i] == 0)
			{
				if (G > PGmax_old)
				{
					active_size--;
					CMath::swap(index[s], index[active_size]);
					s--;
					continue;
				}
				else if (G < 0)
					PG = G;
			}
			else if (alpha[i] == C)
			{
				if (G < PGmin_old)
				{
					active_size--;
					CMath::swap(index[s], index[active_size]);
					s--;
					continue;
				}
				else if (G > 0)
					PG = G;
			}
			else
				PG = G;

			PGmax_new = CMath::max(PGmax_new, PG);
			PGmin_new = CMath::min(PGmin_new, PG);

			if(fabs(PG) > 1.0e-12)
			{
				double alpha_old = alpha[i];
				alpha[i] = CMath::min(CMath::max(alpha[i] - G/QD[i], 0.0), C);
				d = (alpha[i] - alpha_old)*yi;

				prob->x->add_to_dense_vec(d, i, w, n);

				if (prob->use_bias)
					w[n]+=d;
			}
		}

		iter++;
		float64_t gap=PGmax_new - PGmin_new;
		SG_SABS_PROGRESS(gap, -CMath::log10(gap), -CMath::log10(1), -CMath::log10(eps), 6);

		if(gap <= eps)
		{
			if(active_size == l)
				break;
			else
			{
				active_size = l;
				PGmax_old = CMath::INFTY;
				PGmin_old = -CMath::INFTY;
				continue;
			}
		}
		PGmax_old = PGmax_new;
		PGmin_old = PGmin_new;
		if (PGmax_old <= 0)
			PGmax_old = CMath::INFTY;
		if (PGmin_old >= 0)
			PGmin_old = -CMath::INFTY;
	}

	SG_DONE();
	SG_INFO("optimization finished, #iter = %d\n",iter);
	if (iter >= max_iterations)
	{
		SG_WARNING("reaching max number of iterations\nUsing -s 2 may be faster"
				"(also see liblinear FAQ)\n\n");
	}

	// calculate objective value

	double v = 0;
	int nSV = 0;
	for(i=0; i<w_size; i++)
		v += w[i]*w[i];
	for(i=0; i<l; i++)
	{
		v += alpha[i]*(alpha[i]*diag[GETI(i)] - 2);
		if(alpha[i] > 0)
			++nSV;
	}
	SG_INFO("Objective value = %lf\n",v/2);
	SG_INFO("nSV = %d\n",nSV);

	delete [] QD;
	delete [] alpha;
	delete [] y;
	delete [] index;
}

// A coordinate descent algorithm for
// L1-regularized L2-loss support vector classification
//
//  min_w \sum |wj| + C \sum max(0, 1-yi w^T xi)^2,
//
// Given:
// x, y, Cp, Cn
// eps is the stopping tolerance
//
// solution will be put in w

#undef GETI
#define GETI(i) (y[i]+1)
// To support weights for instances, use GETI(i) (i)

void CLibLinear::solve_l1r_l2_svc(
	problem *prob_col, double eps, double Cp, double Cn)
{
	int l = prob_col->l;
	int w_size = prob_col->n;
	int j, s, iter = 0;
	int active_size = w_size;
	int max_num_linesearch = 20;

	double sigma = 0.01;
	double d, G_loss, G, H;
	double Gmax_old = CMath::INFTY;
	double Gmax_new;
	double Gmax_init=0;
	double d_old, d_diff;
	double loss_old=0, loss_new;
	double appxcond, cond;

	int *index = new int[w_size];
	int32_t *y = new int32_t[l];
	double *b = new double[l]; // b = 1-ywTx
	double *xj_sq = new double[w_size];

	CDotFeatures* x = (CDotFeatures*) prob_col->x;
	void* iterator;
	int32_t ind;
	float64_t val;

	double C[3] = {Cn,0,Cp};

	int n = prob_col->n;
	if (prob_col->use_bias)
		n--;

	for(j=0; j<l; j++)
	{
		b[j] = 1;
		if(prob_col->y[j] > 0)
			y[j] = 1;
		else
			y[j] = -1;
	}

	for(j=0; j<w_size; j++)
	{
		w[j] = 0;
		index[j] = j;
		xj_sq[j] = 0;

		if (use_bias && j==n)
		{
			for (ind=0; ind<l; ind++)
				xj_sq[n] += C[GETI(ind)];
		}
		else
		{
			iterator=x->get_feature_iterator(j);
			while (x->get_next_feature(ind, val, iterator))
				xj_sq[j] += C[GETI(ind)]*val*val;
			x->free_feature_iterator(iterator);
		}
	}


	CTime start_time;
	while (iter < max_iterations && !CSignal::cancel_computations())
	{
		if (max_train_time > 0 && start_time.cur_time_diff() > max_train_time)
		  break;

		Gmax_new  = 0;

		for(j=0; j<active_size; j++)
		{
			int i = j+rand()%(active_size-j);
			CMath::swap(index[i], index[j]);
		}

		for(s=0; s<active_size; s++)
		{
			j = index[s];
			G_loss = 0;
			H = 0;

			if (use_bias && j==n)
			{
				for (ind=0; ind<l; ind++)
				{
					if(b[ind] > 0)
					{
						double tmp = C[GETI(ind)]*y[ind];
						G_loss -= tmp*b[ind];
						H += tmp*y[ind];
					}
				}
			}
			else
			{
				iterator=x->get_feature_iterator(j);

				while (x->get_next_feature(ind, val, iterator))
				{
					if(b[ind] > 0)
					{
						double tmp = C[GETI(ind)]*val*y[ind];
						G_loss -= tmp*b[ind];
						H += tmp*val*y[ind];
					}
				}
				x->free_feature_iterator(iterator);
			}

			G_loss *= 2;

			G = G_loss;
			H *= 2;
			H = CMath::max(H, 1e-12);

			double Gp = G+1;
			double Gn = G-1;
			double violation = 0;
			if(w[j] == 0)
			{
				if(Gp < 0)
					violation = -Gp;
				else if(Gn > 0)
					violation = Gn;
				else if(Gp>Gmax_old/l && Gn<-Gmax_old/l)
				{
					active_size--;
					CMath::swap(index[s], index[active_size]);
					s--;
					continue;
				}
			}
			else if(w[j] > 0)
				violation = fabs(Gp);
			else
				violation = fabs(Gn);

			Gmax_new = CMath::max(Gmax_new, violation);

			// obtain Newton direction d
			if(Gp <= H*w[j])
				d = -Gp/H;
			else if(Gn >= H*w[j])
				d = -Gn/H;
			else
				d = -w[j];

			if(fabs(d) < 1.0e-12)
				continue;

			double delta = fabs(w[j]+d)-fabs(w[j]) + G*d;
			d_old = 0;
			int num_linesearch;
			for(num_linesearch=0; num_linesearch < max_num_linesearch; num_linesearch++)
			{
				d_diff = d_old - d;
				cond = fabs(w[j]+d)-fabs(w[j]) - sigma*delta;

				appxcond = xj_sq[j]*d*d + G_loss*d + cond;
				if(appxcond <= 0)
				{
					if (use_bias && j==n)
					{
						for (ind=0; ind<l; ind++)
							b[ind] += d_diff*y[ind];
						break;
					}
					else
					{
						iterator=x->get_feature_iterator(j);
						while (x->get_next_feature(ind, val, iterator))
							b[ind] += d_diff*val*y[ind];

						x->free_feature_iterator(iterator);
						break;
					}
				}

				if(num_linesearch == 0)
				{
					loss_old = 0;
					loss_new = 0;

					if (use_bias && j==n)
					{
						for (ind=0; ind<l; ind++)
						{
							if(b[ind] > 0)
								loss_old += C[GETI(ind)]*b[ind]*b[ind];
							double b_new = b[ind] + d_diff*y[ind];
							b[ind] = b_new;
							if(b_new > 0)
								loss_new += C[GETI(ind)]*b_new*b_new;
						}
					}
					else
					{
						iterator=x->get_feature_iterator(j);
						while (x->get_next_feature(ind, val, iterator))
						{
							if(b[ind] > 0)
								loss_old += C[GETI(ind)]*b[ind]*b[ind];
							double b_new = b[ind] + d_diff*val*y[ind];
							b[ind] = b_new;
							if(b_new > 0)
								loss_new += C[GETI(ind)]*b_new*b_new;
						}
						x->free_feature_iterator(iterator);
					}
				}
				else
				{
					loss_new = 0;
					if (use_bias && j==n)
					{
						for (ind=0; ind<l; ind++)
						{
							double b_new = b[ind] + d_diff*y[ind];
							b[ind] = b_new;
							if(b_new > 0)
								loss_new += C[GETI(ind)]*b_new*b_new;
						}
					}
					else
					{
						iterator=x->get_feature_iterator(j);
						while (x->get_next_feature(ind, val, iterator))
						{
							double b_new = b[ind] + d_diff*val*y[ind];
							b[ind] = b_new;
							if(b_new > 0)
								loss_new += C[GETI(ind)]*b_new*b_new;
						}
						x->free_feature_iterator(iterator);
					}
				}

				cond = cond + loss_new - loss_old;
				if(cond <= 0)
					break;
				else
				{
					d_old = d;
					d *= 0.5;
					delta *= 0.5;
				}
			}

			w[j] += d;

			// recompute b[] if line search takes too many steps
			if(num_linesearch >= max_num_linesearch)
			{
				SG_INFO("#");
				for(int i=0; i<l; i++)
					b[i] = 1;

				for(int i=0; i<n; i++)
				{
					if(w[i]==0)
						continue;

					iterator=x->get_feature_iterator(i);
					while (x->get_next_feature(ind, val, iterator))
						b[ind] -= w[i]*val*y[ind];
					x->free_feature_iterator(iterator);
				}

				if (use_bias && w[n])
				{
					for (ind=0; ind<l; ind++)
						b[ind] -= w[n]*y[ind];
				}
			}
		}

		if(iter == 0)
			Gmax_init = Gmax_new;
		iter++;

		SG_SABS_PROGRESS(Gmax_new, -CMath::log10(Gmax_new), -CMath::log10(Gmax_init), -CMath::log10(eps*Gmax_init), 6);

		if(Gmax_new <= eps*Gmax_init)
		{
			if(active_size == w_size)
				break;
			else
			{
				active_size = w_size;
				Gmax_old = CMath::INFTY;
				continue;
			}
		}

		Gmax_old = Gmax_new;
	}

	SG_DONE();
	SG_INFO("optimization finished, #iter = %d\n", iter);
	if(iter >= max_iterations)
		SG_WARNING("\nWARNING: reaching max number of iterations\n");

	// calculate objective value

	double v = 0;
	int nnz = 0;
	for(j=0; j<w_size; j++)
	{
		if(w[j] != 0)
		{
			v += fabs(w[j]);
			nnz++;
		}
	}
	for(j=0; j<l; j++)
		if(b[j] > 0)
			v += C[GETI(j)]*b[j]*b[j];

	SG_INFO("Objective value = %lf\n", v);
	SG_INFO("#nonzeros/#features = %d/%d\n", nnz, w_size);

	delete [] index;
	delete [] y;
	delete [] b;
	delete [] xj_sq;
}

// A coordinate descent algorithm for
// L1-regularized logistic regression problems
//
//  min_w \sum |wj| + C \sum log(1+exp(-yi w^T xi)),
//
// Given:
// x, y, Cp, Cn
// eps is the stopping tolerance
//
// solution will be put in w

#undef GETI
#define GETI(i) (y[i]+1)
// To support weights for instances, use GETI(i) (i)

void CLibLinear::solve_l1r_lr(
	const problem *prob_col, double eps,
	double Cp, double Cn)
{
	int l = prob_col->l;
	int w_size = prob_col->n;
	int j, s, iter = 0;
	int active_size = w_size;
	int max_num_linesearch = 20;

	double x_min = 0;
	double sigma = 0.01;
	double d, G, H;
	double Gmax_old = CMath::INFTY;
	double Gmax_new;
	double Gmax_init=0;
	double sum1, appxcond1;
	double sum2, appxcond2;
	double cond;

	int *index = new int[w_size];
	int32_t *y = new int32_t[l];
	double *exp_wTx = new double[l];
	double *exp_wTx_new = new double[l];
	double *xj_max = new double[w_size];
	double *C_sum = new double[w_size];
	double *xjneg_sum = new double[w_size];
	double *xjpos_sum = new double[w_size];

	CDotFeatures* x = prob_col->x;
	void* iterator;
	int ind;
	double val;

	double C[3] = {Cn,0,Cp};

	int n = prob_col->n;
	if (prob_col->use_bias)
		n--;

	for(j=0; j<l; j++)
	{
		exp_wTx[j] = 1;
		if(prob_col->y[j] > 0)
			y[j] = 1;
		else
			y[j] = -1;
	}
	for(j=0; j<w_size; j++)
	{
		w[j] = 0;
		index[j] = j;
		xj_max[j] = 0;
		C_sum[j] = 0;
		xjneg_sum[j] = 0;
		xjpos_sum[j] = 0;

		if (use_bias && j==n)
		{
			for (ind=0; ind<l; ind++)
			{
				x_min = CMath::min(x_min, 1.0);
				xj_max[j] = CMath::max(xj_max[j], 1.0);
				C_sum[j] += C[GETI(ind)];
				if(y[ind] == -1)
					xjneg_sum[j] += C[GETI(ind)];
				else
					xjpos_sum[j] += C[GETI(ind)];
			}
		}
		else
		{
			iterator=x->get_feature_iterator(j);
			while (x->get_next_feature(ind, val, iterator))
			{
				x_min = CMath::min(x_min, val);
				xj_max[j] = CMath::max(xj_max[j], val);
				C_sum[j] += C[GETI(ind)];
				if(y[ind] == -1)
					xjneg_sum[j] += C[GETI(ind)]*val;
				else
					xjpos_sum[j] += C[GETI(ind)]*val;
			}
			x->free_feature_iterator(iterator);
		}
	}

	CTime start_time;
	while (iter < max_iterations && !CSignal::cancel_computations())
	{
		if (max_train_time > 0 && start_time.cur_time_diff() > max_train_time)
		  break;

		Gmax_new = 0;

		for(j=0; j<active_size; j++)
		{
			int i = j+rand()%(active_size-j);
			CMath::swap(index[i], index[j]);
		}

		for(s=0; s<active_size; s++)
		{
			j = index[s];
			sum1 = 0;
			sum2 = 0;
			H = 0;

			if (use_bias && j==n)
			{
				for (ind=0; ind<l; ind++)
				{
					double exp_wTxind = exp_wTx[ind];
					double tmp1 = 1.0/(1+exp_wTxind);
					double tmp2 = C[GETI(ind)]*tmp1;
					double tmp3 = tmp2*exp_wTxind;
					sum2 += tmp2;
					sum1 += tmp3;
					H += tmp1*tmp3;
				}
			}
			else
			{
				iterator=x->get_feature_iterator(j);
				while (x->get_next_feature(ind, val, iterator))
				{
					double exp_wTxind = exp_wTx[ind];
					double tmp1 = val/(1+exp_wTxind);
					double tmp2 = C[GETI(ind)]*tmp1;
					double tmp3 = tmp2*exp_wTxind;
					sum2 += tmp2;
					sum1 += tmp3;
					H += tmp1*tmp3;
				}
				x->free_feature_iterator(iterator);
			}

			G = -sum2 + xjneg_sum[j];

			double Gp = G+1;
			double Gn = G-1;
			double violation = 0;
			if(w[j] == 0)
			{
				if(Gp < 0)
					violation = -Gp;
				else if(Gn > 0)
					violation = Gn;
				else if(Gp>Gmax_old/l && Gn<-Gmax_old/l)
				{
					active_size--;
					CMath::swap(index[s], index[active_size]);
					s--;
					continue;
				}
			}
			else if(w[j] > 0)
				violation = fabs(Gp);
			else
				violation = fabs(Gn);

			Gmax_new = CMath::max(Gmax_new, violation);

			// obtain Newton direction d
			if(Gp <= H*w[j])
				d = -Gp/H;
			else if(Gn >= H*w[j])
				d = -Gn/H;
			else
				d = -w[j];

			if(fabs(d) < 1.0e-12)
				continue;

			d = CMath::min(CMath::max(d,-10.0),10.0);

			double delta = fabs(w[j]+d)-fabs(w[j]) + G*d;
			int num_linesearch;
			for(num_linesearch=0; num_linesearch < max_num_linesearch; num_linesearch++)
			{
				cond = fabs(w[j]+d)-fabs(w[j]) - sigma*delta;

				if(x_min >= 0)
				{
					double tmp = exp(d*xj_max[j]);
					appxcond1 = log(1+sum1*(tmp-1)/xj_max[j]/C_sum[j])*C_sum[j] + cond - d*xjpos_sum[j];
					appxcond2 = log(1+sum2*(1/tmp-1)/xj_max[j]/C_sum[j])*C_sum[j] + cond + d*xjneg_sum[j];
					if(CMath::min(appxcond1,appxcond2) <= 0)
					{
						if (use_bias && j==n)
						{
							for (ind=0; ind<l; ind++)
								exp_wTx[ind] *= exp(d);
						}

						else
						{
							iterator=x->get_feature_iterator(j);
							while (x->get_next_feature(ind, val, iterator))
								exp_wTx[ind] *= exp(d*val);
							x->free_feature_iterator(iterator);
						}
						break;
					}
				}

				cond += d*xjneg_sum[j];

				int i = 0;

				if (use_bias && j==n)
				{
					for (ind=0; ind<l; ind++)
					{
						double exp_dx = exp(d);
						exp_wTx_new[i] = exp_wTx[ind]*exp_dx;
						cond += C[GETI(ind)]*log((1+exp_wTx_new[i])/(exp_dx+exp_wTx_new[i]));
						i++;
					}
				}
				else
				{

					iterator=x->get_feature_iterator(j);
					while (x->get_next_feature(ind, val, iterator))
					{
						double exp_dx = exp(d*val);
						exp_wTx_new[i] = exp_wTx[ind]*exp_dx;
						cond += C[GETI(ind)]*log((1+exp_wTx_new[i])/(exp_dx+exp_wTx_new[i]));
						i++;
					}
					x->free_feature_iterator(iterator);
				}

				if(cond <= 0)
				{
					i = 0;
					if (use_bias && j==n)
					{
						for (ind=0; ind<l; ind++)
						{
							exp_wTx[ind] = exp_wTx_new[i];
							i++;
						}
					}
					else
					{
						iterator=x->get_feature_iterator(j);
						while (x->get_next_feature(ind, val, iterator))
						{
							exp_wTx[ind] = exp_wTx_new[i];
							i++;
						}
						x->free_feature_iterator(iterator);
					}
					break;
				}
				else
				{
					d *= 0.5;
					delta *= 0.5;
				}
			}

			w[j] += d;

			// recompute exp_wTx[] if line search takes too many steps
			if(num_linesearch >= max_num_linesearch)
			{
				SG_INFO("#");
				for(int i=0; i<l; i++)
					exp_wTx[i] = 0;

				for(int i=0; i<w_size; i++)
				{
					if(w[i]==0) continue;

					if (use_bias && i==n)
					{
						for (ind=0; ind<l; ind++)
							exp_wTx[ind] += w[i];
					}
					else
					{
						iterator=x->get_feature_iterator(i);
						while (x->get_next_feature(ind, val, iterator))
							exp_wTx[ind] += w[i]*val;
						x->free_feature_iterator(iterator);
					}
				}

				for(int i=0; i<l; i++)
					exp_wTx[i] = exp(exp_wTx[i]);
			}
		}

		if(iter == 0)
			Gmax_init = Gmax_new;
		iter++;
		SG_SABS_PROGRESS(Gmax_new, -CMath::log10(Gmax_new), -CMath::log10(Gmax_init), -CMath::log10(eps*Gmax_init), 6);

		if(Gmax_new <= eps*Gmax_init)
		{
			if(active_size == w_size)
				break;
			else
			{
				active_size = w_size;
				Gmax_old = CMath::INFTY;
				continue;
			}
		}

		Gmax_old = Gmax_new;
	}

	SG_DONE();
	SG_INFO("optimization finished, #iter = %d\n", iter);
	if(iter >= max_iterations)
		SG_WARNING("\nWARNING: reaching max number of iterations\n");

	// calculate objective value

	double v = 0;
	int nnz = 0;
	for(j=0; j<w_size; j++)
		if(w[j] != 0)
		{
			v += fabs(w[j]);
			nnz++;
		}
	for(j=0; j<l; j++)
		if(y[j] == 1)
			v += C[GETI(j)]*log(1+1/exp_wTx[j]);
		else
			v += C[GETI(j)]*log(1+exp_wTx[j]);

	SG_INFO("Objective value = %lf\n", v);
	SG_INFO("#nonzeros/#features = %d/%d\n", nnz, w_size);

	delete [] index;
	delete [] y;
	delete [] exp_wTx;
	delete [] exp_wTx_new;
	delete [] xj_max;
	delete [] C_sum;
	delete [] xjneg_sum;
	delete [] xjpos_sum;
}

void CLibLinear::get_linear_term(float64_t** linear_term, int32_t* len)
{
	if (!m_linear_term_len || !m_linear_term)
		SG_ERROR("Please assign linear term first!\n");

	*linear_term=(float64_t*) SG_MALLOC(sizeof(float64_t)*m_linear_term_len);

	for (int32_t i=0; i<m_linear_term_len; i++)
		(*linear_term)[i]=m_linear_term[i];
}

void CLibLinear::init_linear_term()
{
	if (!labels)
		SG_ERROR("Please assign labels first!\n");

	delete[] m_linear_term;

	m_linear_term_len=labels->get_num_labels();
	m_linear_term = new float64_t[m_linear_term_len];
	CMath::fill_vector(m_linear_term, m_linear_term_len, -1.0);
}

#endif //HAVE_LAPACK
