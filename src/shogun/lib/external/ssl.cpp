/*    Copyright 2006 Vikas Sindhwani (vikass@cs.uchicago.edu)
	  SVM-lin: Fast SVM Solvers for Supervised and Semi-supervised Learning

	  This file is part of SVM-lin.

	  SVM-lin is free software; you can redistribute it and/or modify
	  it under the terms of the GNU General Public License as published by
	  the Free Software Foundation; either version 2 of the License, or
	  (at your option) any later version.

	  SVM-lin is distributed in the hope that it will be useful,
	  but WITHOUT ANY WARRANTY; without even the implied warranty of
	  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
	  GNU General Public License for more details.

	  You should have received a copy of the GNU General Public License
	  along with SVM-lin (see gpl.txt); if not, write to the Free Software
	  Foundation, Inc., 51 Franklin St, Fifth Floor, Boston, MA  02110-1301  USA
	  */

#include <stdio.h>
#include <stdlib.h>
#include <ctype.h>
#include <algorithm>

#include <shogun/io/SGIO.h>
#include <shogun/mathematics/Math.h>
#include <shogun/features/SparseFeatures.h>
#include <shogun/lib/external/ssl.h>

namespace shogun
{
void ssl_train(struct data *Data,
		struct options *Options,
		struct vector_double *Weights,
		struct vector_double *Outputs)
{
	// initialize
	initialize(Weights,Data->n,0.0);
	initialize(Outputs,Data->m,0.0);
	vector_int    *Subset  = SG_MALLOC(vector_int, 1);
	initialize(Subset,Data->m);
	// call the right algorithm
	int32_t optimality = 0;
	switch(Options->algo)
	{
		case -1:
			SG_SINFO("Regularized Least Squares Regression (CGLS)\n")
			optimality=CGLS(Data,Options,Subset,Weights,Outputs);
			break;
		case RLS:
			SG_SINFO("Regularized Least Squares Classification (CGLS)\n")
			optimality=CGLS(Data,Options,Subset,Weights,Outputs);
			break;
		case SVM:
			SG_SINFO("Modified Finite Newton L2-SVM (L2-SVM-MFN)\n")
			optimality=L2_SVM_MFN(Data,Options,Weights,Outputs,0);
			break;
		case TSVM:
			SG_SINFO("Transductive L2-SVM (TSVM)\n")
			optimality=TSVM_MFN(Data,Options,Weights,Outputs);
			break;
		case DA_SVM:
			SG_SINFO("Deterministic Annealing Semi-supervised L2-SVM (DAS3VM)\n")
			optimality=DA_S3VM(Data,Options,Weights,Outputs);
			break;
		default:
			SG_SERROR("Algorithm unspecified\n")
	}

	if (!optimality)
		SG_SWARNING("SSL-Algorithm terminated without reaching optimum.\n")

	SG_FREE(Subset->vec);
    SG_FREE(Subset);
	return;
}

int32_t CGLS(
	const struct data *Data, const struct options *Options,
	const struct vector_int *Subset, struct vector_double *Weights,
	struct vector_double *Outputs)
{
	SG_SDEBUG("CGLS starting...")

	/* Disassemble the structures */
	int32_t active = Subset->d;
	int32_t *J = Subset->vec;
	CDotFeatures* features=Data->features;
	float64_t *Y = Data->Y;
	float64_t *C = Data->C;
	int32_t n  = Data->n;
	float64_t lambda = Options->lambda;
	int32_t cgitermax = Options->cgitermax;
	float64_t epsilon = Options->epsilon;
	float64_t *beta = Weights->vec;
	float64_t *o  = Outputs->vec;
	// initialize z
	float64_t *z = SG_MALLOC(float64_t, active);
	float64_t *q = SG_MALLOC(float64_t, active);
	int32_t ii=0;
	for (int32_t i = active ; i-- ;){
		ii=J[i];
		z[i]  = C[ii]*(Y[ii] - o[ii]);
	}
	float64_t *r = SG_MALLOC(float64_t, n);
	for (int32_t i = n ; i-- ;)
		r[i] = 0.0;
	for (register int32_t j=0; j < active; j++)
	{
		features->add_to_dense_vec(z[j], J[j], r, n-1);
		r[n-1]+=Options->bias*z[j]; //bias (modelled as last dim)
	}
	float64_t *p = SG_MALLOC(float64_t, n);
	float64_t omega1 = 0.0;
	for (int32_t i = n ; i-- ;)
	{
		r[i] -= lambda*beta[i];
		p[i] = r[i];
		omega1 += r[i]*r[i];
	}
	float64_t omega_p = omega1;
	float64_t omega_q = 0.0;
	float64_t inv_omega2 = 1/omega1;
	float64_t scale = 0.0;
	float64_t omega_z=0.0;
	float64_t gamma = 0.0;
	int32_t cgiter = 0;
	int32_t optimality = 0;
	float64_t epsilon2 = epsilon*epsilon;
	// iterate
	while(cgiter < cgitermax)
	{
		cgiter++;
		omega_q=0.0;
		float64_t t=0.0;
		register int32_t i,j;
		// #pragma omp parallel for private(i,j)
		for (i=0; i < active; i++)
		{
			ii=J[i];
			t=features->dense_dot(ii, p, n-1);
			t+=Options->bias*p[n-1]; //bias (modelled as last dim)
			q[i]=t;
			omega_q += C[ii]*t*t;
		}
		gamma = omega1/(lambda*omega_p + omega_q);
		inv_omega2 = 1/omega1;
		for (i = n ; i-- ;)
		{
			r[i] = 0.0;
			beta[i] += gamma*p[i];
		}
		omega_z=0.0;
		for (i = active ; i-- ;)
		{
			ii=J[i];
			o[ii] += gamma*q[i];
			z[i] -= gamma*C[ii]*q[i];
			omega_z+=z[i]*z[i];
		}
		for (j=0; j < active; j++)
		{
			t=z[j];

			features->add_to_dense_vec(t, J[j], r, n-1);
			r[n-1]+=Options->bias*t; //bias (modelled as last dim)
		}
		omega1 = 0.0;
		for (i = n ; i-- ;)
		{
			r[i] -= lambda*beta[i];
			omega1 += r[i]*r[i];
		}
		if(omega1 < epsilon2*omega_z)
		{
			optimality=1;
			break;
		}
		omega_p=0.0;
		scale=omega1*inv_omega2;
		for(i = n ; i-- ;)
		{
			p[i] = r[i] + p[i]*scale;
			omega_p += p[i]*p[i];
		}
	}
	SG_SDEBUG("...Done.")
	SG_SINFO("CGLS converged in %d iteration(s)", cgiter)

	SG_FREE(z);
	SG_FREE(q);
	SG_FREE(r);
	SG_FREE(p);
	return optimality;
}

int32_t L2_SVM_MFN(
	const struct data *Data, struct options *Options,
	struct vector_double *Weights, struct vector_double *Outputs,
	int32_t ini)
{
	/* Disassemble the structures */
	CDotFeatures* features=Data->features;
	float64_t *Y = Data->Y;
	float64_t *C = Data->C;
	int32_t n  = Data->n;
	int32_t m  = Data->m;
	float64_t lambda = Options->lambda;
	float64_t epsilon;
	float64_t *w = Weights->vec;
	float64_t *o = Outputs->vec;
	float64_t F_old = 0.0;
	float64_t F = 0.0;
	float64_t diff=0.0;
	vector_int *ActiveSubset = SG_MALLOC(vector_int, 1);
	ActiveSubset->vec = SG_MALLOC(int32_t, m);
	ActiveSubset->d = m;
	// initialize
	if(ini==0) {
		epsilon=BIG_EPSILON;
		Options->cgitermax=SMALL_CGITERMAX;
		Options->epsilon=BIG_EPSILON;
	}
	else {epsilon = Options->epsilon;}
	for (int32_t i=0;i<n;i++) F+=w[i]*w[i];
	F=0.5*lambda*F;
	int32_t active=0;
	int32_t inactive=m-1; // l-1
	for (int32_t i=0; i<m ; i++)
	{
		diff=1-Y[i]*o[i];
		if(diff>0)
		{
			ActiveSubset->vec[active]=i;
			active++;
			F+=0.5*C[i]*diff*diff;
		}
		else
		{
			ActiveSubset->vec[inactive]=i;
			inactive--;
		}
	}
	ActiveSubset->d=active;
	int32_t iter=0;
	int32_t opt=0;
	int32_t opt2=0;
	vector_double *Weights_bar = SG_MALLOC(vector_double, 1);
	vector_double *Outputs_bar = SG_MALLOC(vector_double, 1);
	float64_t *w_bar = SG_MALLOC(float64_t, n);
	float64_t *o_bar = SG_MALLOC(float64_t, m);
	Weights_bar->vec=w_bar;
	Outputs_bar->vec=o_bar;
	Weights_bar->d=n;
	Outputs_bar->d=m;
	float64_t delta=0.0;
	float64_t t=0.0;
	int32_t ii = 0;
	while(iter<MFNITERMAX)
	{
		iter++;
		SG_SDEBUG("L2_SVM_MFN Iteration# %d (%d active examples, objective_value = %f)\n", iter, active, F)
		for (int32_t i=n; i-- ;)
			w_bar[i]=w[i];
		for (int32_t i=m; i-- ;)
			o_bar[i]=o[i];

		opt=CGLS(Data,Options,ActiveSubset,Weights_bar,Outputs_bar);
		for(register int32_t i=active; i < m; i++)
		{
			ii=ActiveSubset->vec[i];

			t=features->dense_dot(ii, w_bar, n-1);
			t+=Options->bias*w_bar[n-1]; //bias (modelled as last dim)

			o_bar[ii]=t;
		}
		if(ini==0) {Options->cgitermax=CGITERMAX; ini=1;};
		opt2=1;
		for (int32_t i=0;i<m;i++)
		{
			ii=ActiveSubset->vec[i];
			if(i<active)
				opt2=(opt2 && (Y[ii]*o_bar[ii]<=1+epsilon));
			else
				opt2=(opt2 && (Y[ii]*o_bar[ii]>=1-epsilon));
			if(opt2==0) break;
		}
		if(opt && opt2) // l
		{
			if(epsilon==BIG_EPSILON)
			{
				epsilon=EPSILON;
				Options->epsilon=EPSILON;
				SG_SDEBUG("epsilon = %f case converged (speedup heuristic 2). Continuing with epsilon=%f",  BIG_EPSILON , EPSILON)
				continue;
			}
			else
			{
				for (int32_t i=n; i-- ;)
					w[i]=w_bar[i];
				for (int32_t i=m; i-- ;)
					o[i]=o_bar[i];
				SG_FREE(ActiveSubset->vec);
				SG_FREE(ActiveSubset);
				SG_FREE(o_bar);
				SG_FREE(w_bar);
				SG_FREE(Weights_bar);
				SG_FREE(Outputs_bar);
				SG_SINFO("L2_SVM_MFN converged (optimality) in %d", iter)
				return 1;
			}
		}
		delta=line_search(w,w_bar,lambda,o,o_bar,Y,C,n,m);
		SG_SDEBUG("LINE_SEARCH delta = %f\n", delta)
		F_old=F;
		F=0.0;
		for (int32_t i=n; i-- ;) {
			w[i]+=delta*(w_bar[i]-w[i]);
			F+=w[i]*w[i];
		}
		F=0.5*lambda*F;
		active=0;
		inactive=m-1;
		for (int32_t i=0; i<m ; i++)
		{
			o[i]+=delta*(o_bar[i]-o[i]);
			diff=1-Y[i]*o[i];
			if(diff>0)
			{
				ActiveSubset->vec[active]=i;
				active++;
				F+=0.5*C[i]*diff*diff;
			}
			else
			{
				ActiveSubset->vec[inactive]=i;
				inactive--;
			}
		}
		ActiveSubset->d=active;
		if(CMath::abs(F-F_old)<RELATIVE_STOP_EPS*CMath::abs(F_old))
		{
			SG_FREE(ActiveSubset->vec);
			SG_FREE(ActiveSubset);
			SG_FREE(o_bar);
			SG_FREE(w_bar);
			SG_FREE(Weights_bar);
			SG_FREE(Outputs_bar);
			SG_SINFO("L2_SVM_MFN converged (rel. criterion) in %d iterations", iter)
			return 2;
		}
	}
	SG_FREE(ActiveSubset->vec);
	SG_FREE(ActiveSubset);
	SG_FREE(o_bar);
	SG_FREE(w_bar);
	SG_FREE(Weights_bar);
	SG_FREE(Outputs_bar);
	SG_SINFO("L2_SVM_MFN converged (max iter exceeded) in %d iterations", iter)
	return 0;
}

float64_t line_search(float64_t *w,
		float64_t *w_bar,
		float64_t lambda,
		float64_t *o,
		float64_t *o_bar,
		float64_t *Y,
		float64_t *C,
		int32_t d, /* data dimensionality -- 'n' */
		int32_t l) /* number of examples */
{
	float64_t omegaL = 0.0;
	float64_t omegaR = 0.0;
	float64_t diff=0.0;
	for(int32_t i=d; i--; )
	{
		diff=w_bar[i]-w[i];
		omegaL+=w[i]*diff;
		omegaR+=w_bar[i]*diff;
	}
	omegaL=lambda*omegaL;
	omegaR=lambda*omegaR;
	float64_t L=0.0;
	float64_t R=0.0;
	int32_t ii=0;
	for (int32_t i=0;i<l;i++)
	{
		if(Y[i]*o[i]<1)
		{
			diff=C[i]*(o_bar[i]-o[i]);
			L+=(o[i]-Y[i])*diff;
			R+=(o_bar[i]-Y[i])*diff;
		}
	}
	L+=omegaL;
	R+=omegaR;
	Delta* deltas=SG_MALLOC(Delta, l);
	int32_t p=0;
	for(int32_t i=0;i<l;i++)
	{
		diff=Y[i]*(o_bar[i]-o[i]);

		if(Y[i]*o[i]<1)
		{
			if(diff>0)
			{
				deltas[p].delta=(1-Y[i]*o[i])/diff;
				deltas[p].index=i;
				deltas[p].s=-1;
				p++;
			}
		}
		else
		{
			if(diff<0)
			{
				deltas[p].delta=(1-Y[i]*o[i])/diff;
				deltas[p].index=i;
				deltas[p].s=1;
				p++;
			}
		}
	}
	std::sort(deltas,deltas+p);
	float64_t delta_prime=0.0;
	for (int32_t i=0;i<p;i++)
	{
		delta_prime = L + deltas[i].delta*(R-L);
		if(delta_prime>=0)
			break;
		ii=deltas[i].index;
		diff=(deltas[i].s)*C[ii]*(o_bar[ii]-o[ii]);
		L+=diff*(o[ii]-Y[ii]);
		R+=diff*(o_bar[ii]-Y[ii]);
	}
	SG_FREE(deltas);
	return (-L/(R-L));
}

int32_t TSVM_MFN(
	const struct data *Data, struct options *Options,
	struct vector_double *Weights, struct vector_double *Outputs)
{
	/* Setup labeled-only examples and train L2_SVM_MFN */
	struct data *Data_Labeled = SG_MALLOC(data, 1);
	struct vector_double *Outputs_Labeled = SG_MALLOC(vector_double, 1);
	initialize(Outputs_Labeled,Data->l,0.0);
	SG_SDEBUG("Initializing weights, unknown labels")
	GetLabeledData(Data_Labeled,Data); /* gets labeled data and sets C=1/l */
	L2_SVM_MFN(Data_Labeled, Options, Weights,Outputs_Labeled,0);
	///FIXME Clear(Data_Labeled);
	/* Use this weight vector to classify R*u unlabeled examples as
	   positive*/
	int32_t p=0,q=0;
	float64_t t=0.0;
	int32_t *JU = SG_MALLOC(int32_t, Data->u);
	float64_t *ou = SG_MALLOC(float64_t, Data->u);
	float64_t lambda_0 = TSVM_LAMBDA_SMALL;
	for (int32_t i=0;i<Data->m;i++)
	{
		if(Data->Y[i]==0.0)
		{
			t=Data->features->dense_dot(i, Weights->vec, Data->n-1);
			t+=Options->bias*Weights->vec[Data->n-1]; //bias (modelled as last dim)

			Outputs->vec[i]=t;
			Data->C[i]=lambda_0*1.0/Data->u;
			JU[q]=i;
			ou[q]=t;
			q++;
		}
		else
		{
			Outputs->vec[i]=Outputs_Labeled->vec[p];
			Data->C[i]=1.0/Data->l;
			p++;
		}
	}
	std::nth_element(ou,ou+int32_t((1-Options->R)*Data->u-1),ou+Data->u);
	float64_t thresh=*(ou+int32_t((1-Options->R)*Data->u)-1);
	SG_FREE(ou);
	for (int32_t i=0;i<Data->u;i++)
	{
		if(Outputs->vec[JU[i]]>thresh)
			Data->Y[JU[i]]=1.0;
		else
			Data->Y[JU[i]]=-1.0;
	}
	for (int32_t i=0;i<Data->n;i++)
		Weights->vec[i]=0.0;
	for (int32_t i=0;i<Data->m;i++)
		Outputs->vec[i]=0.0;
	L2_SVM_MFN(Data,Options,Weights,Outputs,0);
	int32_t num_switches=0;
	int32_t s=0;
	int32_t last_round=0;
	while(lambda_0 <= Options->lambda_u)
	{
		int32_t iter2=0;
		while(1){
			s=switch_labels(Data->Y,Outputs->vec,JU,Data->u,Options->S);
			if(s==0) break;
			iter2++;
			SG_SDEBUG("****** lambda_0 = %f iteration = %d ************************************\n", lambda_0, iter2)
			SG_SDEBUG("Optimizing unknown labels. switched %d labels.\n")
			num_switches+=s;
			SG_SDEBUG("Optimizing weights\n")
			L2_SVM_MFN(Data,Options,Weights,Outputs,1);
		}
		if(last_round==1) break;
		lambda_0=TSVM_ANNEALING_RATE*lambda_0;
		if(lambda_0 >= Options->lambda_u) {lambda_0 = Options->lambda_u; last_round=1;}
		for (int32_t i=0;i<Data->u;i++)
			Data->C[JU[i]]=lambda_0*1.0/Data->u;
		SG_SDEBUG("****** lambda0 increased to %f%% of lambda_u = %f ************************\n", lambda_0*100/Options->lambda_u, Options->lambda_u)
		SG_SDEBUG("Optimizing weights\n")
		L2_SVM_MFN(Data,Options,Weights,Outputs,1);
	}
	SG_SDEBUG("Total Number of Switches = %d\n", num_switches)
	/* reset labels */
	for (int32_t i=0;i<Data->u;i++) Data->Y[JU[i]] = 0.0;
	float64_t F = transductive_cost(norm_square(Weights),Data->Y,Outputs->vec,Outputs->d,Options->lambda,Options->lambda_u);
	SG_SDEBUG("Objective Value = %f\n",F)
	delete [] JU;
	return num_switches;
}

int32_t switch_labels(float64_t* Y, float64_t* o, int32_t* JU, int32_t u, int32_t S)
{
	int32_t npos=0;
	int32_t nneg=0;
	for (int32_t i=0;i<u;i++)
	{
		if((Y[JU[i]]>0) && (o[JU[i]]<1.0)) npos++;
		if((Y[JU[i]]<0) && (-o[JU[i]]<1.0)) nneg++;
	}
	Delta* positive=SG_MALLOC(Delta, npos);
	Delta* negative=SG_MALLOC(Delta, nneg);
	int32_t p=0;
	int32_t n=0;
	int32_t ii=0;
	for (int32_t i=0;i<u;i++)
	{
		ii=JU[i];
		if((Y[ii]>0.0) && (o[ii]<1.0)) {
			positive[p].delta=o[ii];
			positive[p].index=ii;
			positive[p].s=0;
			p++;};
			if((Y[ii]<0.0) && (-o[ii]<1.0))
			{
				negative[n].delta=-o[ii];
				negative[n].index=ii;
				negative[n].s=0;
				n++;};
	}
	std::sort(positive,positive+npos);
	std::sort(negative,negative+nneg);
	int32_t s=-1;
	while(1)
	{
		s++;
		if((s>=S) || (positive[s].delta>=-negative[s].delta) || (s>=npos) || (s>=nneg))
			break;
		Y[positive[s].index]=-1.0;
		Y[negative[s].index]= 1.0;
	}
	SG_FREE(positive);
	SG_FREE(negative);
	return s;
}

int32_t DA_S3VM(
	struct data *Data, struct options *Options, struct vector_double *Weights,
	struct vector_double *Outputs)
{
	float64_t T = DA_INIT_TEMP*Options->lambda_u;
	int32_t iter1 = 0, iter2 =0;
	float64_t *p = SG_MALLOC(float64_t, Data->u);
	float64_t *q = SG_MALLOC(float64_t, Data->u);
	float64_t *g = SG_MALLOC(float64_t, Data->u);
	float64_t F,F_min;
	float64_t *w_min = SG_MALLOC(float64_t, Data->n);
	float64_t *o_min = SG_MALLOC(float64_t, Data->m);
	float64_t *w = Weights->vec;
	float64_t *o = Outputs->vec;
	float64_t kl_divergence = 1.0;
	/*initialize */
	SG_SDEBUG("Initializing weights, p")
	for (int32_t i=0;i<Data->u; i++)
		p[i] = Options->R;
	/* record which examples are unlabeled */
	int32_t *JU = SG_MALLOC(int32_t, Data->u);
	int32_t j=0;
	for(int32_t i=0;i<Data->m;i++)
	{
		if(Data->Y[i]==0.0)
		{JU[j]=i;j++;}
	}
	float64_t H = entropy(p,Data->u);
	optimize_w(Data,p,Options,Weights,Outputs,0);
	F = transductive_cost(norm_square(Weights),Data->Y,Outputs->vec,Outputs->d,Options->lambda,Options->lambda_u);
	F_min = F;
	for (int32_t i=0;i<Weights->d;i++)
		w_min[i]=w[i];
	for (int32_t i=0;i<Outputs->d;i++)
		o_min[i]=o[i];
	while((iter1 < DA_OUTER_ITERMAX) && (H > Options->epsilon))
	{
		iter1++;
		iter2=0;
		kl_divergence=1.0;
		while((iter2 < DA_INNER_ITERMAX) && (kl_divergence > Options->epsilon))
		{
			iter2++;
			for (int32_t i=0;i<Data->u;i++)
			{
				q[i]=p[i];
				g[i] = Options->lambda_u*((o[JU[i]] > 1 ? 0 : (1 - o[JU[i]])*(1 - o[JU[i]])) - (o[JU[i]]< -1 ? 0 : (1 + o[JU[i]])*(1 + o[JU[i]])));
			}
			SG_SDEBUG("Optimizing p.\n")
			optimize_p(g,Data->u,T,Options->R,p);
			kl_divergence=KL(p,q,Data->u);
			SG_SDEBUG("Optimizing weights\n")
			optimize_w(Data,p,Options,Weights,Outputs,1);
			F = transductive_cost(norm_square(Weights),Data->Y,Outputs->vec,Outputs->d,Options->lambda,Options->lambda_u);
			if(F < F_min)
			{
				F_min = F;
				for (int32_t i=0;i<Weights->d;i++)
					w_min[i]=w[i];
				for (int32_t i=0;i<Outputs->d;i++)
					o_min[i]=o[i];
			}
			SG_SDEBUG("***** outer_iter = %d  T = %g  inner_iter = %d  kl = %g  cost = %g *****\n",iter1,T,iter2,kl_divergence,F)
		}
		H = entropy(p,Data->u);
		SG_SDEBUG("***** Finished outer_iter = %d T = %g  Entropy = %g ***\n", iter1,T,H)
		T = T/DA_ANNEALING_RATE;
	}
	for (int32_t i=0;i<Weights->d;i++)
		w[i]=w_min[i];
	for (int32_t i=0;i<Outputs->d;i++)
		o[i]=o_min[i];
	/* may want to reset the original Y */
	SG_FREE(p);
	SG_FREE(q);
	SG_FREE(g);
	SG_FREE(JU);
	SG_FREE(w_min);
	SG_FREE(o_min);
	SG_SINFO("(min) Objective Value = %f", F_min)
	return 1;
}

int32_t optimize_w(
	const struct data *Data, const float64_t *p, struct options *Options,
	struct vector_double *Weights, struct vector_double *Outputs, int32_t ini)
{
	int32_t i,j;
	CDotFeatures* features=Data->features;
	int32_t n  = Data->n;
	int32_t m  = Data->m;
	int32_t u  = Data->u;
	float64_t lambda = Options->lambda;
	float64_t epsilon;
	float64_t *w = Weights->vec;
	float64_t *o = SG_MALLOC(float64_t, m+u);
	float64_t *Y = SG_MALLOC(float64_t, m+u);
	float64_t *C = SG_MALLOC(float64_t, m+u);
	int32_t *labeled_indices = SG_MALLOC(int32_t, m);
	float64_t F_old = 0.0;
	float64_t F = 0.0;
	float64_t diff=0.0;
	float64_t lambda_u_by_u = Options->lambda_u/u;
	vector_int *ActiveSubset = SG_MALLOC(vector_int, 1);
	ActiveSubset->vec = SG_MALLOC(int32_t, m);
	ActiveSubset->d = m;
	// initialize
	if(ini==0)
	{
		epsilon=BIG_EPSILON;
		Options->cgitermax=SMALL_CGITERMAX;
		Options->epsilon=BIG_EPSILON;}
	else {epsilon = Options->epsilon;}

	for(i=0;i<n;i++) F+=w[i]*w[i];
	F=lambda*F;
	int32_t active=0;
	int32_t inactive=m-1; // l-1
	float64_t temp1;
	float64_t temp2;

	j = 0;
	for(i=0; i<m ; i++)
	{
		o[i]=Outputs->vec[i];
		if(Data->Y[i]==0.0)
		{
			labeled_indices[i]=0;
			o[m+j]=o[i];
			Y[i]=1;
			Y[m+j]=-1;
			C[i]=lambda_u_by_u*p[j];
			C[m+j]=lambda_u_by_u*(1-p[j]);
			ActiveSubset->vec[active]=i;
			active++;
			diff = 1 - CMath::abs(o[i]);
			if(diff>0)
			{
				Data->Y[i] = 2*p[j]-1;
				Data->C[i] = lambda_u_by_u;
				temp1 = (1 - o[i]);
				temp2 = (1 + o[i]);
				F+=lambda_u_by_u*(p[j]*temp1*temp1 + (1-p[j])*temp2*temp2);
			}
			else
			{
				if(o[i]>0)
				{
					Data->Y[i] = -1.0;
					Data->C[i] = C[m+j];
				}
				else
				{
					Data->Y[i] = 1.0;
					Data->C[i] = C[i];
				}
				temp1 = (1-Data->Y[i]*o[i]);
				F+= Data->C[i]*temp1*temp1;
			}
			j++;
		}
		else
		{
			labeled_indices[i]=1;
			Y[i]=Data->Y[i];
			C[i]=1.0/Data->l;
			Data->C[i]=1.0/Data->l;
			diff=1-Data->Y[i]*o[i];
			if(diff>0)
			{
				ActiveSubset->vec[active]=i;
				active++;
				F+=Data->C[i]*diff*diff;
			}
			else
			{
				ActiveSubset->vec[inactive]=i;
				inactive--;
			}
		}
	}
	F=0.5*F;
	ActiveSubset->d=active;
	int32_t iter=0;
	int32_t opt=0;
	int32_t opt2=0;
	vector_double *Weights_bar = SG_MALLOC(vector_double, 1);
	vector_double *Outputs_bar = SG_MALLOC(vector_double, 1);
	float64_t *w_bar = SG_MALLOC(float64_t, n);
	float64_t *o_bar = SG_MALLOC(float64_t, m+u);
	Weights_bar->vec=w_bar;
	Outputs_bar->vec=o_bar;
	Weights_bar->d=n;
	Outputs_bar->d=m; /* read only the top m ; bottom u will be copies */
	float64_t delta=0.0;
	float64_t t=0.0;
	int32_t ii = 0;
	while(iter<MFNITERMAX)
	{
		iter++;
		SG_SDEBUG("L2_SVM_MFN Iteration# %d (%d active examples,  objective_value = %f)", iter, active, F)
		for(i=n; i-- ;)
			w_bar[i]=w[i];

		for(i=m+u; i-- ;)
			o_bar[i]=o[i];
		opt=CGLS(Data,Options,ActiveSubset,Weights_bar,Outputs_bar);
		for(i=active; i < m; i++)
		{
			ii=ActiveSubset->vec[i];
			t=features->dense_dot(ii, w_bar, n-1);
			t+=Options->bias*w_bar[n-1]; //bias (modelled as last dim)

			o_bar[ii]=t;
		}
		// make o_bar consistent in the bottom half
		j=0;
		for(i=0; i<m;i++)
		{
			if(labeled_indices[i]==0)
			{o_bar[m+j]=o_bar[i]; j++;};
		}
		if(ini==0) {Options->cgitermax=CGITERMAX; ini=1;};
		opt2=1;
		for(i=0; i < m ;i++)
		{
			ii=ActiveSubset->vec[i];
			if(i<active)
			{
				if(labeled_indices[ii]==1)
					opt2=(opt2 && (Data->Y[ii]*o_bar[ii]<=1+epsilon));
				else
				{
					if(CMath::abs(o[ii])<1)
						opt2=(opt2 && (CMath::abs(o_bar[ii])<=1+epsilon));
					else
						opt2=(opt2 && (CMath::abs(o_bar[ii])>=1-epsilon));
				}
			}
			else
				opt2=(opt2 && (Data->Y[ii]*o_bar[ii]>=1-epsilon));
			if(opt2==0) break;
		}
		if(opt && opt2) // l
		{
			if(epsilon==BIG_EPSILON)
			{
				epsilon=EPSILON;
				Options->epsilon=EPSILON;
				SG_SDEBUG("epsilon = %f case converged (speedup heuristic 2). Continuing with epsilon=%f\n", BIG_EPSILON, EPSILON)
				continue;
			}
			else
			{
				for(i=n; i-- ;)
					w[i]=w_bar[i];
				for(i=m; i-- ;)
					Outputs->vec[i]=o_bar[i];
				for(i=m; i-- ;)
				{
					if(labeled_indices[i]==0)
						Data->Y[i]=0.0;
				}
				SG_FREE(ActiveSubset->vec);
				SG_FREE(ActiveSubset);
				SG_FREE(o_bar);
				SG_FREE(w_bar);
				SG_FREE(o);
				SG_FREE(Weights_bar);
				SG_FREE(Outputs_bar);
				SG_FREE(Y);
				SG_FREE(C);
				SG_FREE(labeled_indices);
				SG_SINFO("L2_SVM_MFN converged in %d iteration(s)", iter)
				return 1;
			}
		}

		delta=line_search(w,w_bar,lambda,o,o_bar,Y,C,n,m+u);
		SG_SDEBUG("LINE_SEARCH delta = %f", delta)
		F_old=F;
		F=0.0;
		for(i=0;i<n;i++) {w[i]+=delta*(w_bar[i]-w[i]);  F+=w[i]*w[i];}
		F=lambda*F;
		j=0;
		active=0;
		inactive=m-1;
		for(i=0; i<m ; i++)
		{
			o[i]+=delta*(o_bar[i]-o[i]);
			if(labeled_indices[i]==0)
			{
				o[m+j]=o[i];
				ActiveSubset->vec[active]=i;
				active++;
				diff = 1 - CMath::abs(o[i]);
				if(diff>0)
				{
					Data->Y[i] = 2*p[j]-1;
					Data->C[i] = lambda_u_by_u;
					temp1 = (1 - o[i]);
					temp2 = (1 + o[i]);
					F+=lambda_u_by_u*(p[j]*temp1*temp1 + (1-p[j])*temp2*temp2);
				}
				else
				{
					if(o[i]>0)
					{
						Data->Y[i] = -1;
						Data->C[i] = C[m+j];
					}
					else
					{
						Data->Y[i] = +1;
						Data->C[i] = C[i];
					}
					temp1=(1-Data->Y[i]*o[i]);
					F+= Data->C[i]*temp1*temp1;
				}
				j++;
			}
			else
			{
				diff=1-Data->Y[i]*o[i];
				if(diff>0)
				{
					ActiveSubset->vec[active]=i;
					active++;
					F+=Data->C[i]*diff*diff;
				}
				else
				{
					ActiveSubset->vec[inactive]=i;
					inactive--;
				}
			}
		}
		F=0.5*F;
		ActiveSubset->d=active;
		if(CMath::abs(F-F_old)<EPSILON)
			break;
	}
	for(i=m; i-- ;)
	{
		Outputs->vec[i]=o[i];
		if(labeled_indices[i]==0)
			Data->Y[i]=0.0;
	}
	SG_FREE(ActiveSubset->vec);
	SG_FREE(labeled_indices);
	SG_FREE(ActiveSubset);
	SG_FREE(o_bar);
	SG_FREE(w_bar);
	SG_FREE(o);
	SG_FREE(Weights_bar);
	SG_FREE(Outputs_bar);
	SG_FREE(Y);
	SG_FREE(C);
	SG_SINFO("L2_SVM_MFN converged in %d iterations", iter)
	return 0;
}

void optimize_p(
	const float64_t* g, int32_t u, float64_t T, float64_t r, float64_t* p)
{
	int32_t iter=0;
	float64_t epsilon=1e-10;
	int32_t maxiter=500;
	float64_t nu_minus=g[0];
	float64_t nu_plus=g[0];
	for (int32_t i=0;i<u;i++)
	{
		if(g[i]<nu_minus) nu_minus=g[i];
		if(g[i]>nu_plus) nu_plus=g[i];
	};

	float64_t b=T*log((1-r)/r);
	nu_minus-=b;
	nu_plus-=b;
	float64_t nu=(nu_plus+nu_minus)/2;
	float64_t Bnu=0.0;
	float64_t BnuPrime=0.0;
	float64_t s=0.0;
	float64_t tmp=0.0;
	for (int32_t i=0;i<u;i++)
	{
		s=exp((g[i]-nu)/T);
		if(!(CMath::is_infinity(s)))
		{
			tmp=1.0/(1.0+s);
			Bnu+=tmp;
			BnuPrime+=s*tmp*tmp;
		}
	}
	Bnu=Bnu/u;
	Bnu-=r;
	BnuPrime=BnuPrime/(T*u);
	float64_t nuHat=0.0;
	while((CMath::abs(Bnu)>epsilon) && (iter < maxiter))
	{
		iter++;
		if(CMath::abs(BnuPrime)>0.0)
			nuHat=nu-Bnu/BnuPrime;
		if((CMath::abs(BnuPrime) > 0.0) | (nuHat>nu_plus)  | (nuHat < nu_minus))
			nu=(nu_minus+nu_plus)/2.0;
		else
			nu=nuHat;
		Bnu=0.0;
		BnuPrime=0.0;
		for(int32_t i=0;i<u;i++)
		{
			s=exp((g[i]-nu)/T);
			if(!(CMath::is_infinity(s)))
			{
				tmp=1.0/(1.0+s);
				Bnu+=tmp;
				BnuPrime+=s*tmp*tmp;
			}
		}
		Bnu=Bnu/u;
		Bnu-=r;
		BnuPrime=BnuPrime/(T*u);
		if(Bnu<0)
			nu_minus=nu;
		else
			nu_plus=nu;
		if(CMath::abs(nu_minus-nu_plus)<epsilon)
			break;
	}
	if(CMath::abs(Bnu)>epsilon)
		SG_SWARNING("Warning (Root): root not found to required precision\n")

	for (int32_t i=0;i<u;i++)
	{
		s=exp((g[i]-nu)/T);
		if(CMath::is_infinity(s)) p[i]=0.0;
		else p[i]=1.0/(1.0+s);
	}
	SG_SINFO(" root (nu) = %f B(nu) = %f", nu, Bnu)
}

float64_t transductive_cost(
	float64_t normWeights, float64_t *Y, float64_t *Outputs, int32_t m,
	float64_t lambda, float64_t lambda_u)
{
	float64_t F1=0.0,F2=0.0, o=0.0, y=0.0;
	int32_t u=0,l=0;
	for (int32_t i=0;i<m;i++)
	{
		o=Outputs[i];
		y=Y[i];
		if(y==0.0)
		{F1 += CMath::abs(o) > 1 ? 0 : (1 - CMath::abs(o))*(1 - CMath::abs(o)); u++;}
		else
		{F2 += y*o > 1 ? 0 : (1-y*o)*(1-y*o); l++;}
	}
	float64_t F;
	F = 0.5*(lambda*normWeights + lambda_u*F1/u + F2/l);
	return F;
}

float64_t entropy(const float64_t *p, int32_t u)
{
	float64_t h=0.0;
	float64_t q=0.0;
	for (int32_t i=0;i<u;i++)
	{
		q=p[i];
		if(q>0 && q<1)
			h+= -(q*CMath::log2(q) + (1-q)*CMath::log2(1-q));
	}
	return h/u;
}

float64_t KL(const float64_t *p, const float64_t *q, int32_t u)
{
	float64_t h=0.0;
	float64_t p1=0.0;
	float64_t q1=0.0;
	float64_t g=0.0;
	for (int32_t i=0;i<u;i++)
	{
		p1=p[i];
		q1=q[i];
		if(p1>1-1e-8) p1-=1e-8;
		if(p1<1-1e-8) p1+=1e-8;
		if(q1>1-1e-8) q1-=1e-8;
		if(q1<1-1e-8) q1+=1e-8;
		g= (p1*CMath::log2(p1/q1) + (1-p1)*CMath::log2((1-p1)/(1-q1)));
		if(CMath::abs(g)<1e-12 || CMath::is_nan(g)) g=0.0;
		h+=g;
	}
	return h/u;
}

/********************** UTILITIES ********************/
float64_t norm_square(const vector_double *A)
{
	float64_t x=0.0, t=0.0;
	for(int32_t i=0;i<A->d;i++)
	{
		t=A->vec[i];
		x+=t*t;
	}
	return x;
}

void initialize(struct vector_double *A, int32_t k, float64_t a)
{
	float64_t *vec = SG_MALLOC(float64_t, k);
	for (int32_t i=0;i<k;i++)
		vec[i]=a;
	A->vec = vec;
	A->d   = k;
	return;
}

void initialize(struct vector_int *A, int32_t k)
{
	int32_t *vec = SG_MALLOC(int32_t, k);
	for(int32_t i=0;i<k;i++)
		vec[i]=i;
	A->vec = vec;
	A->d   = k;
	return;
}

void GetLabeledData(struct data *D, const struct data *Data)
{
	/*FIXME
	int32_t *J = SG_MALLOC(int, Data->l);
	D->C   = SG_MALLOC(float64_t, Data->l);
	D->Y   = SG_MALLOC(float64_t, Data->l);
	int32_t nz=0;
	int32_t k=0;
	int32_t rowptrs_=Data->l;
	for(int32_t i=0;i<Data->m;i++)
	{
		if(Data->Y[i]!=0.0)
		{
			J[k]=i;
			D->Y[k]=Data->Y[i];
			D->C[k]=1.0/Data->l;
			nz+=(Data->rowptr[i+1] - Data->rowptr[i]);
			k++;
		}
	}
	D->val    = SG_MALLOC(float64_t, nz);
	D->colind = SG_MALLOC(int32_t, nz);
	D->rowptr = new int32_trowptrs_+1];
	nz=0;
	for(int32_t i=0;i<Data->l;i++)
	{
		D->rowptr[i]=nz;
		for(int32_t j=Data->rowptr[J[i]];j<Data->rowptr[J[i]+1];j++)
		{
			D->val[nz] = Data->val[j];
			D->colind[nz] = Data->colind[j];
			nz++;
		}
	}
	D->rowptr[rowptrs_]=nz;
	D->nz=nz;
	D->l=Data->l;
	D->m=Data->l;
	D->n=Data->n;
	D->u=0;
	SG_FREE(J);*/
}
}
