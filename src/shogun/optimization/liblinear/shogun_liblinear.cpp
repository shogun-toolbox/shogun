/*
 * Copyright (c) 2007-2009 The LIBLINEAR Project.
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *
 * 1. Redistributions of source code must retain the above copyright
 * notice, this list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright
 * notice, this list of conditions and the following disclaimer in the
 * documentation and/or other materials provided with the distribution.
 *
 * 3. Neither name of copyright holders nor the names of its contributors
 * may be used to endorse or promote products derived from this software
 * without specific prior written permission.
 *
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
 * ``AS IS'' AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
 * LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
 * A PARTICULAR PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE REGENTS OR
 * CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 * EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 * PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
 * PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
 * LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
 * NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 * SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */
#include <shogun/lib/config.h>
#ifndef DOXYGEN_SHOULD_SKIP_THIS
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdarg.h>

#include <shogun/mathematics/Math.h>
#include <shogun/optimization/liblinear/shogun_liblinear.h>
#include <shogun/optimization/liblinear/tron.h>
#include <shogun/lib/Time.h>
#include <shogun/lib/Signal.h>

using namespace shogun;

l2r_lr_fun::l2r_lr_fun(const liblinear_problem *p, float64_t* Cs)
{
	int l=p->l;

	this->m_prob = p;

	z = SG_MALLOC(double, l);
	D = SG_MALLOC(double, l);
	C = Cs;
}

l2r_lr_fun::~l2r_lr_fun()
{
	SG_FREE(z);
	SG_FREE(D);
}


double l2r_lr_fun::fun(double *w)
{
	int i;
	double f=0;
	float64_t *y=m_prob->y;
	int l=m_prob->l;
	int32_t n=m_prob->n;

	Xv(w, z);
	for(i=0;i<l;i++)
	{
		double yz = y[i]*z[i];
		if (yz >= 0)
			f += C[i]*log(1 + exp(-yz));
		else
			f += C[i]*(-yz+log(1 + exp(yz)));
	}
	f += 0.5 *SGVector<float64_t>::dot(w,w,n);

	return(f);
}

void l2r_lr_fun::grad(double *w, double *g)
{
	int i;
	float64_t *y=m_prob->y;
	int l=m_prob->l;
	int w_size=get_nr_variable();

	for(i=0;i<l;i++)
	{
		z[i] = 1/(1 + exp(-y[i]*z[i]));
		D[i] = z[i]*(1-z[i]);
		z[i] = C[i]*(z[i]-1)*y[i];
	}
	XTv(z, g);

	for(i=0;i<w_size;i++)
		g[i] = w[i] + g[i];
}

int l2r_lr_fun::get_nr_variable()
{
	return m_prob->n;
}

void l2r_lr_fun::Hv(double *s, double *Hs)
{
	int i;
	int l=m_prob->l;
	int w_size=get_nr_variable();
	double *wa = SG_MALLOC(double, l);

	Xv(s, wa);
	for(i=0;i<l;i++)
		wa[i] = C[i]*D[i]*wa[i];

	XTv(wa, Hs);
	for(i=0;i<w_size;i++)
		Hs[i] = s[i] + Hs[i];
	SG_FREE(wa);
}

void l2r_lr_fun::Xv(double *v, double *res_Xv)
{
	int32_t l=m_prob->l;
	int32_t n=m_prob->n;
	float64_t bias=0;

	if (m_prob->use_bias)
	{
		n--;
		bias=v[n];
	}

	m_prob->x->dense_dot_range(res_Xv, 0, l, NULL, v, n, bias);
}

void l2r_lr_fun::XTv(double *v, double *res_XTv)
{
	int l=m_prob->l;
	int32_t n=m_prob->n;

	memset(res_XTv, 0, sizeof(double)*m_prob->n);

	if (m_prob->use_bias)
		n--;

	for (int32_t i=0;i<l;i++)
	{
		m_prob->x->add_to_dense_vec(v[i], i, res_XTv, n);

		if (m_prob->use_bias)
			res_XTv[n]+=v[i];
	}
}

l2r_l2_svc_fun::l2r_l2_svc_fun(const liblinear_problem *p, double* Cs)
{
	int l=p->l;

	this->m_prob = p;

	z = SG_MALLOC(double, l);
	D = SG_MALLOC(double, l);
	I = SG_MALLOC(int, l);
	C=Cs;

}

l2r_l2_svc_fun::~l2r_l2_svc_fun()
{
	SG_FREE(z);
	SG_FREE(D);
	SG_FREE(I);
}

double l2r_l2_svc_fun::fun(double *w)
{
	int i;
	double f=0;
	float64_t *y=m_prob->y;
	int l=m_prob->l;
	int w_size=get_nr_variable();

	Xv(w, z);
	for(i=0;i<l;i++)
	{
		z[i] = y[i]*z[i];
		double d = 1-z[i];
		if (d > 0)
			f += C[i]*d*d;
	}
	f += 0.5*SGVector<float64_t>::dot(w, w, w_size);

	return(f);
}

void l2r_l2_svc_fun::grad(double *w, double *g)
{
	int i;
	float64_t *y=m_prob->y;
	int l=m_prob->l;
	int w_size=get_nr_variable();

	sizeI = 0;
	for (i=0;i<l;i++)
		if (z[i] < 1)
		{
			z[sizeI] = C[i]*y[i]*(z[i]-1);
			I[sizeI] = i;
			sizeI++;
		}
	subXTv(z, g);

	for(i=0;i<w_size;i++)
		g[i] = w[i] + 2*g[i];
}

int l2r_l2_svc_fun::get_nr_variable()
{
	return m_prob->n;
}

void l2r_l2_svc_fun::Hv(double *s, double *Hs)
{
	int i;
	int l=m_prob->l;
	int w_size=get_nr_variable();
	double *wa = SG_MALLOC(double, l);

	subXv(s, wa);
	for(i=0;i<sizeI;i++)
		wa[i] = C[I[i]]*wa[i];

	subXTv(wa, Hs);
	for(i=0;i<w_size;i++)
		Hs[i] = s[i] + 2*Hs[i];
	SG_FREE(wa);
}

void l2r_l2_svc_fun::Xv(double *v, double *res_Xv)
{
	int32_t l=m_prob->l;
	int32_t n=m_prob->n;
	float64_t bias=0;

	if (m_prob->use_bias)
	{
		n--;
		bias=v[n];
	}

	m_prob->x->dense_dot_range(res_Xv, 0, l, NULL, v, n, bias);
}

void l2r_l2_svc_fun::subXv(double *v, double *res_Xv)
{
	int32_t n=m_prob->n;
	float64_t bias=0;

	if (m_prob->use_bias)
	{
		n--;
		bias=v[n];
	}

	m_prob->x->dense_dot_range_subset(I, sizeI, res_Xv, NULL, v, n, bias);

	/*for (int32_t i=0;i<sizeI;i++)
	{
		res_Xv[i]=m_prob->x->dense_dot(I[i], v, n);

		if (m_prob->use_bias)
			res_Xv[i]+=bias;
	}*/
}

void l2r_l2_svc_fun::subXTv(double *v, double *XTv)
{
	int32_t n=m_prob->n;

	if (m_prob->use_bias)
		n--;

	memset(XTv, 0, sizeof(float64_t)*m_prob->n);
	for (int32_t i=0;i<sizeI;i++)
	{
		m_prob->x->add_to_dense_vec(v[i], I[i], XTv, n);

		if (m_prob->use_bias)
			XTv[n]+=v[i];
	}
}

l2r_l2_svr_fun::l2r_l2_svr_fun(const liblinear_problem *prob, double *Cs, double p):
	l2r_l2_svc_fun(prob, Cs)
{
	m_p = p;
}

double l2r_l2_svr_fun::fun(double *w)
{
	int i;
	double f=0;
	double *y=m_prob->y;
	int l=m_prob->l;
	int w_size=get_nr_variable();
	double d;

	Xv(w, z);

	for(i=0;i<w_size;i++)
		f += w[i]*w[i];
	f /= 2;
	for(i=0;i<l;i++)
	{
		d = z[i] - y[i];
		if(d < -m_p)
			f += C[i]*(d+m_p)*(d+m_p);
		else if(d > m_p)
			f += C[i]*(d-m_p)*(d-m_p);
	}

	return(f);
}

void l2r_l2_svr_fun::grad(double *w, double *g)
{
	int i;
	double *y=m_prob->y;
	int l=m_prob->l;
	int w_size=get_nr_variable();
	double d;

	sizeI = 0;
	for(i=0;i<l;i++)
	{
		d = z[i] - y[i];

		// generate index set I
		if(d < -m_p)
		{
			z[sizeI] = C[i]*(d+m_p);
			I[sizeI] = i;
			sizeI++;
		}
		else if(d > m_p)
		{
			z[sizeI] = C[i]*(d-m_p);
			I[sizeI] = i;
			sizeI++;
		}

	}
	subXTv(z, g);

	for(i=0;i<w_size;i++)
		g[i] = w[i] + 2*g[i];
}


// A coordinate descent algorithm for
// multi-class support vector machines by Crammer and Singer
//
//  min_{\alpha}  0.5 \sum_m ||w_m(\alpha)||^2 + \sum_i \sum_m e^m_i alpha^m_i
//    s.t.     \alpha^m_i <= C^m_i \forall m,i , \sum_m \alpha^m_i=0 \forall i
//
//  where e^m_i = 0 if y_i  = m,
//        e^m_i = 1 if y_i != m,
//  C^m_i = C if m  = y_i,
//  C^m_i = 0 if m != y_i,
//  and w_m(\alpha) = \sum_i \alpha^m_i x_i
//
// Given:
// x, y, C
// eps is the stopping tolerance
//
// solution will be put in w

#define GETI(i) (prob->y[i])
// To support weights for instances, use GETI(i) (i)

Solver_MCSVM_CS::Solver_MCSVM_CS(const liblinear_problem *p, int n_class,
                                 double *weighted_C, double *w0_reg,
                                 double epsilon, int max_it, double max_time,
                                 mcsvm_state* given_state)
{
	this->w_size = p->n;
	this->l = p->l;
	this->nr_class = n_class;
	this->eps = epsilon;
	this->max_iter = max_it;
	this->prob = p;
	this->C = weighted_C;
	this->w0 = w0_reg;
	this->max_train_time = max_time;
	this->state = given_state;
}

Solver_MCSVM_CS::~Solver_MCSVM_CS()
{
}

int compare_double(const void *a, const void *b)
{
	if(*(double *)a > *(double *)b)
		return -1;
	if(*(double *)a < *(double *)b)
		return 1;
	return 0;
}

void Solver_MCSVM_CS::solve_sub_problem(double A_i, int yi, double C_yi, int active_i, double *alpha_new)
{
	int r;
	double *D=SGVector<float64_t>::clone_vector(state->B, active_i);

	if(yi < active_i)
		D[yi] += A_i*C_yi;
	qsort(D, active_i, sizeof(double), compare_double);

	double beta = D[0] - A_i*C_yi;
	for(r=1;r<active_i && beta<r*D[r];r++)
		beta += D[r];

	beta /= r;
	for(r=0;r<active_i;r++)
	{
		if(r == yi)
			alpha_new[r] = CMath::min(C_yi, (beta-state->B[r])/A_i);
		else
			alpha_new[r] = CMath::min((double)0, (beta - state->B[r])/A_i);
	}
	SG_FREE(D);
}

bool Solver_MCSVM_CS::be_shrunk(int i, int m, int yi, double alpha_i, double minG)
{
	double bound = 0;
	if(m == yi)
		bound = C[int32_t(GETI(i))];
	if(alpha_i == bound && state->G[m] < minG)
		return true;
	return false;
}

void Solver_MCSVM_CS::solve()
{
	int i, m, s, k;
	int iter = 0;
	double *w,*B,*G,*alpha,*alpha_new,*QD,*d_val;
	int *index,*d_ind,*alpha_index,*y_index,*active_size_i;

	if (!state->allocated)
	{
		state->w = SG_CALLOC(double, nr_class*w_size);
		state->B = SG_CALLOC(double, nr_class);
		state->G = SG_CALLOC(double, nr_class);
		state->alpha = SG_CALLOC(double, l*nr_class);
		state->alpha_new = SG_CALLOC(double, nr_class);
		state->index = SG_CALLOC(int, l);
		state->QD = SG_CALLOC(double, l);
		state->d_ind = SG_CALLOC(int, nr_class);
		state->d_val = SG_CALLOC(double, nr_class);
		state->alpha_index = SG_CALLOC(int, nr_class*l);
		state->y_index = SG_CALLOC(int, l);
		state->active_size_i = SG_CALLOC(int, l);
		state->allocated = true;
	}
	w = state->w;
	B = state->B;
	G = state->G;
	alpha = state->alpha;
	alpha_new = state->alpha_new;
	index = state->index;
	QD = state->QD;
	d_ind = state->d_ind;
	d_val = state->d_val;
	alpha_index = state->alpha_index;
	y_index = state->y_index;
	active_size_i = state->active_size_i;

	double* tx = SG_MALLOC(double, w_size);
	int dim = prob->x->get_dim_feature_space();

	int active_size = l;
	double eps_shrink = CMath::max(10.0*eps, 1.0); // stopping tolerance for shrinking
	bool start_from_all = true;
	CTime start_time;
	// initial
	if (!state->inited)
	{
		for(i=0;i<l;i++)
		{
			for(m=0;m<nr_class;m++)
				alpha_index[i*nr_class+m] = m;

			QD[i] = prob->x->dot(i, prob->x,i);
			if (prob->use_bias)
				QD[i] += 1.0;

			active_size_i[i] = nr_class;
			y_index[i] = prob->y[i];
			index[i] = i;
		}
		state->inited = true;
	}

	while(iter < max_iter && !CSignal::cancel_computations())
	{
		double stopping = -CMath::INFTY;
		for(i=0;i<active_size;i++)
		{
			int j = CMath::random(i, active_size-1);
			CMath::swap(index[i], index[j]);
		}
		for(s=0;s<active_size;s++)
		{
			i = index[s];
			double Ai = QD[i];
			double *alpha_i = &alpha[i*nr_class];
			int *alpha_index_i = &alpha_index[i*nr_class];

			if(Ai > 0)
			{
				for(m=0;m<active_size_i[i];m++)
					G[m] = 1;
				if(y_index[i] < active_size_i[i])
					G[y_index[i]] = 0;

				memset(tx,0,dim*sizeof(double));
				prob->x->add_to_dense_vec(1.0,i,tx,dim);
				for (k=0; k<dim; k++)
				{
					if (tx[k]==0.0)
						continue;

					double* w_i = &w[k*nr_class];
					for (m=0; m<active_size_i[i]; m++)
						G[m] += w_i[alpha_index_i[m]]*tx[k];
				}

				// experimental
				// ***
				if (prob->use_bias)
				{
					double *w_i = &w[(w_size-1)*nr_class];
					for(m=0; m<active_size_i[i]; m++)
						G[m] += w_i[alpha_index_i[m]];
				}
				if (w0)
				{
					for (k=0; k<dim; k++)
					{
						double *w0_i = &w0[k*nr_class];
						for(m=0; m<active_size_i[i]; m++)
							G[m] += w0_i[alpha_index_i[m]];
					}
				}
				// ***

				double minG = CMath::INFTY;
				double maxG = -CMath::INFTY;
				for(m=0;m<active_size_i[i];m++)
				{
					if(alpha_i[alpha_index_i[m]] < 0 && G[m] < minG)
						minG = G[m];
					if(G[m] > maxG)
						maxG = G[m];
				}
				if(y_index[i] < active_size_i[i])
					if(alpha_i[int32_t(prob->y[i])] < C[int32_t(GETI(i))] && G[y_index[i]] < minG)
						minG = G[y_index[i]];

				for(m=0;m<active_size_i[i];m++)
				{
					if(be_shrunk(i, m, y_index[i], alpha_i[alpha_index_i[m]], minG))
					{
						active_size_i[i]--;
						while(active_size_i[i]>m)
						{
							if(!be_shrunk(i, active_size_i[i], y_index[i],
											alpha_i[alpha_index_i[active_size_i[i]]], minG))
							{
								CMath::swap(alpha_index_i[m], alpha_index_i[active_size_i[i]]);
								CMath::swap(G[m], G[active_size_i[i]]);
								if(y_index[i] == active_size_i[i])
									y_index[i] = m;
								else if(y_index[i] == m)
									y_index[i] = active_size_i[i];
								break;
							}
							active_size_i[i]--;
						}
					}
				}

				if(active_size_i[i] <= 1)
				{
					active_size--;
					CMath::swap(index[s], index[active_size]);
					s--;
					continue;
				}

				if(maxG-minG <= 1e-12)
					continue;
				else
					stopping = CMath::CMath::max(maxG - minG, stopping);

				for(m=0;m<active_size_i[i];m++)
					B[m] = G[m] - Ai*alpha_i[alpha_index_i[m]] ;

				solve_sub_problem(Ai, y_index[i], C[int32_t(GETI(i))], active_size_i[i], alpha_new);
				int nz_d = 0;
				for(m=0;m<active_size_i[i];m++)
				{
					double d = alpha_new[m] - alpha_i[alpha_index_i[m]];
					alpha_i[alpha_index_i[m]] = alpha_new[m];
					if(fabs(d) >= 1e-12)
					{
						d_ind[nz_d] = alpha_index_i[m];
						d_val[nz_d] = d;
						nz_d++;
					}
				}

				memset(tx,0,dim*sizeof(double));
				prob->x->add_to_dense_vec(1.0,i,tx,dim);
				for (k=0; k<dim; k++)
				{
					if (tx[k]==0.0)
						continue;

					double* w_i = &w[k*nr_class];
					for (m=0; m<nz_d; m++)
						w_i[d_ind[m]] += d_val[m]*tx[k];
				}
				// experimental
				// ***
				if (prob->use_bias)
				{
					double *w_i = &w[(w_size-1)*nr_class];
					for(m=0;m<nz_d;m++)
						w_i[d_ind[m]] += d_val[m];
				}
				// ***
			}
		}

		iter++;
		/*
		if(iter % 10 == 0)
		{
			SG_SINFO(".")
		}
		*/

		if(stopping < eps_shrink)
		{
			if(stopping < eps && start_from_all == true)
				break;
			else
			{
				active_size = l;
				for(i=0;i<l;i++)
					active_size_i[i] = nr_class;
				//SG_SINFO("*")
				eps_shrink = CMath::max(eps_shrink/2, eps);
				start_from_all = true;
			}
		}
		else
			start_from_all = false;

		if (max_train_time!=0.0 && max_train_time < start_time.cur_time_diff())
			break;
	}

	SG_SINFO("\noptimization finished, #iter = %d\n",iter)
	if (iter >= max_iter)
		SG_SINFO("Warning: reaching max number of iterations\n")

	SG_FREE(tx);
}

//
// Interface functions
//

void destroy_model(struct liblinear_model *model_)
{
	SG_FREE(model_->w);
	SG_FREE(model_->label);
	SG_FREE(model_);
}

void destroy_param(liblinear_parameter* param)
{
	SG_FREE(param->weight_label);
	SG_FREE(param->weight);
}
#endif // DOXYGEN_SHOULD_SKIP_THIS
