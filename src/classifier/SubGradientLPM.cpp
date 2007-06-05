/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 2 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2007 Soeren Sonnenburg
 * Written (W) 2007 Vojtech Franc 
 * Copyright (C) 2007 Fraunhofer Institute FIRST and Max-Planck-Society
 */

#include "lib/config.h"
#include "lib/Mathematics.h"
#include "lib/Signal.h"
#include "lib/Time.h"
#include "classifier/SparseLinearClassifier.h"
#include "classifier/SubGradientLPM.h"
#include "classifier/svm/qpbsvmlib.h"
#include "features/SparseFeatures.h"
#include "features/Labels.h"

#define DEBUG_SUBGRADIENTLPM

extern double sparsity;
double lpmtim;

CSubGradientLPM::CSubGradientLPM() : CSparseLinearClassifier(), C1(1), C2(1), epsilon(1e-5), qpsize(42), qpsize_max(2000), use_bias(false), delta_active(0), delta_bound(0)
{
}

CSubGradientLPM::CSubGradientLPM(DREAL C, CSparseFeatures<DREAL>* traindat, CLabels* trainlab) 
: CSparseLinearClassifier(), C1(C), C2(C), epsilon(1e-5), qpsize(42), qpsize_max(2000),
	use_bias(false), delta_active(0), delta_bound(0)
{
	CSparseLinearClassifier::features=traindat;
	CClassifier::labels=trainlab;
}


CSubGradientLPM::~CSubGradientLPM()
{
}

INT CSubGradientLPM::find_active(INT num_feat, INT num_vec, INT& num_active, INT& num_bound)
{
	delta_active=0;
	num_active=0;
	num_bound=0;

	for (INT i=0; i<num_vec; i++)
	{
		active[i]=0;

		//within margin/wrong side
		if (proj[i] < 1-work_epsilon)
		{
			idx_active[num_active++]=i;
			active[i]=1;
		}

		//on margin
		if (CMath::abs(proj[i]-1) <= work_epsilon)
		{
			idx_bound[num_bound++]=i;
			active[i]=2;
		}

		if (active[i]!=old_active[i])
			delta_active++;
	}

	pos_idx=0;
	neg_idx=0;
	zero_idx=0;

	for (INT i=0; i<num_feat; i++)
	{
		if (w[i]>work_epsilon)
		{
			w_pos[pos_idx++]=i;
			grad_w[i]=1;
		}
		else if (w[i]<-work_epsilon)
		{
			w_neg[neg_idx++]=i;
			grad_w[i]=-1;
		}

		if (CMath::abs(w[i])<=work_epsilon)
		{
			w_zero[zero_idx++]=i;
			grad_w[i]=-1;
		}
	}

	return delta_active;
}


void CSubGradientLPM::update_active(INT num_feat, INT num_vec)

{
	for (INT i=0; i<num_vec; i++)
	{
		if (active[i]==1 && old_active[i]!=1)
		{
			features->add_to_dense_vec(C1*get_label(i), i, sum_CXy_active, num_feat);
			if (use_bias)
				sum_Cy_active+=C1*get_label(i);
		}
		else if (old_active[i]==1 && active[i]!=1)
		{
			features->add_to_dense_vec(-C1*get_label(i), i, sum_CXy_active, num_feat);
			if (use_bias)
				sum_Cy_active-=C1*get_label(i);
		}
	}

	CMath::swap(active,old_active);
}

DREAL CSubGradientLPM::line_search(INT num_feat, INT num_vec)
{
	INT num_hinge=0;
	DREAL alpha=0;
	DREAL sgrad=0;

	DREAL* A=new DREAL[num_feat+num_vec];
	DREAL* B=new DREAL[num_feat+num_vec];
	DREAL* C=new DREAL[num_feat+num_vec];
	DREAL* D=new DREAL[num_feat+num_vec];

	for (INT i=0; i<num_feat+num_vec; i++)
	{
		if (i<num_feat)
		{
			A[i]=-grad_w[i];
			B[i]=w[i];
			C[i]=+grad_w[i];
			D[i]=-w[i];
		}
		else
		{
			DREAL p=get_label(i-num_feat)*features->dense_dot(1.0, i-num_feat, grad_w, num_feat, grad_b);
			grad_proj[i-num_feat]=p;
			
			A[i]=0;
			B[i]=0;
			C[i]=C1*p;
			D[i]=C1*(1-proj[i-num_feat]);
		}

		if (A[i]==C[i] && B[i]>D[i])
			sgrad+=A[i]+C[i];
		else if (A[i]==C[i] && B[i]==D[i])
			sgrad+=CMath::max(A[i],C[i]);
		else if (A[i]!=C[i])
		{
			hinge_point[num_hinge]=(D[i]-B[i])/(A[i]-C[i]);
			hinge_idx[num_hinge]=i; // index into A,B,C,D arrays
			num_hinge++;

			if (A[i]>C[i])
				sgrad+=C[i];
			if (A[i]<C[i])
				sgrad+=A[i];
		}
	}

	SG_PRINT("sgrad:%f\n", sgrad);
	CMath::display_vector(A, num_feat+num_vec, "A");
	CMath::display_vector(B, num_feat+num_vec, "B");
	CMath::display_vector(C, num_feat+num_vec, "C");
	CMath::display_vector(D, num_feat+num_vec, "D");
	CMath::display_vector(hinge_point, num_feat+num_vec, "hinge_point");
	CMath::display_vector(hinge_idx, num_feat+num_vec, "hinge_idx");
	//ASSERT(0);

	CMath::qsort(hinge_point, hinge_idx, num_hinge);
	CMath::display_vector(hinge_point, num_feat+num_vec, "hinge_point_sorted");


	INT i=-1;
	while (i < num_hinge-1 && sgrad < 0)
	{
		i+=1;

		if (A[hinge_idx[i]] > C[hinge_idx[i]])
			sgrad += A[hinge_idx[i]] - C[hinge_idx[i]];
		else
			sgrad += C[hinge_idx[i]] - A[hinge_idx[i]];
	}

	alpha = hinge_point[i];

	delete[] D;
	delete[] C;
	delete[] B;
	delete[] A;

	SG_PRINT("alpha=%f\n", alpha);
	return alpha;
}

DREAL CSubGradientLPM::compute_min_subgradient(INT num_feat, INT num_vec, INT num_active, INT num_bound)
{
	DREAL dir_deriv=0;
	solver->init(E_QP);

	if (zero_idx+num_bound > 0)
	{
		SG_PRINT("num_var:%d (zero:%d, bound:%d) num_feat:%d\n", zero_idx+num_bound, zero_idx,num_bound, num_feat);
		CMath::display_vector(grad_w, num_feat+1, "grad_w");
		CMath::add(grad_w, 1.0, grad_w, -1.0, sum_CXy_active, num_feat);
		grad_w[num_feat]= -sum_Cy_active;
		grad_b = -sum_Cy_active;

		CMath::display_vector(grad_w, num_feat+1, "grad_w");

		solver->setup_subgradientlpm_QP(C1, get_labels(), get_features(), idx_bound, num_bound,
				w_zero, zero_idx,
				grad_w, num_feat+1,
				use_bias);

		solver->optimize(beta);
		CMath::display_vector(beta, 5, "beta");
		for (INT i=0; i<zero_idx+num_bound; i++)
			beta[i]=beta[i+num_feat+1];

		CMath::display_vector(beta, zero_idx+num_bound, "beta");

		for (INT i=0; i<zero_idx+num_bound; i++)
		{
			if (i<zero_idx)
				grad_w[w_zero[i]]+=beta[w_zero[i]];
			else
			{
				features->add_to_dense_vec(-C1*beta[i]*get_label(idx_bound[i-zero_idx]), idx_bound[i-zero_idx], grad_w, num_feat);
				if (use_bias)
					grad_b -=  C1 * get_label(idx_bound[i-zero_idx])*beta[i-zero_idx];
			}
		}

		CMath::display_vector(w_zero, zero_idx, "w_zero");
		CMath::display_vector(grad_w, num_feat, "grad_w");
		SG_PRINT("grad_b=%f\n", grad_b);
	}
	else
	{
		CMath::add(grad_w, 1.0, w, -1.0, sum_CXy_active, num_feat);
		grad_b = -sum_Cy_active;

		//dir_deriv = CMath::dot(grad_w, grad_w, num_feat)+ grad_b*grad_b;
	}

	solver->cleanup();


	SG_PRINT("Gradient   : |subgrad_W|^2=%f, |subgrad_b|^2=%f\n",
			CMath::dot(grad_w, grad_w, num_feat), grad_b*grad_b);

	return dir_deriv;
}

DREAL CSubGradientLPM::compute_objective(INT num_feat, INT num_vec)
{
	DREAL result= CMath::sum_abs(w, num_feat);
	
	for (INT i=0; i<num_vec; i++)
	{
		if (proj[i]<1.0)
			result += C1 * (1.0-proj[i]);
	}

	return result;
}

void CSubGradientLPM::compute_projection(INT num_feat, INT num_vec)
{
	for (INT i=0; i<num_vec; i++)
		proj[i]=get_label(i)*features->dense_dot(1.0, i, w, num_feat, bias);
}

void CSubGradientLPM::update_projection(DREAL alpha, INT num_vec)
{
	CMath::vec1_plus_scalar_times_vec2(proj,-alpha, grad_proj, num_vec);
}

void CSubGradientLPM::init(INT num_vec, INT num_feat)
{
	// alloc normal and bias inited with 0
	delete[] w;
	w=new DREAL[num_feat];
	ASSERT(w);
	for (INT i=0; i<num_feat; i++)
		w[i]=0.0;
	//CMath::random_vector(w, num_feat, -1.0, 1.0);
	bias=0;
	num_it_noimprovement=0;
	grad_b=0;
	set_w(w, num_feat);
	qpsize_limit=5000;

	w_pos=new INT[num_feat];
	ASSERT(w_pos);
	memset(w_pos,0,sizeof(INT)*num_feat);
	
	w_zero=new INT[num_feat];
	ASSERT(w_zero);
	memset(w_zero,0,sizeof(INT)*num_feat);
	
	w_neg=new INT[num_feat];
	ASSERT(w_neg);
	memset(w_neg,0,sizeof(INT)*num_feat);

	grad_w=new DREAL[num_feat+1];
	ASSERT(grad_w);
	memset(grad_w,0,sizeof(DREAL)*(num_feat+1));

	sum_CXy_active=new DREAL[num_feat];
	ASSERT(sum_CXy_active);
	memset(sum_CXy_active,0,sizeof(DREAL)*num_feat);

	sum_Cy_active=0;

	proj= new DREAL[num_vec];
	ASSERT(proj);
	memset(proj,0,sizeof(DREAL)*num_vec);

	tmp_proj=new DREAL[num_vec];
	ASSERT(proj);
	memset(proj,0,sizeof(DREAL)*num_vec);

	tmp_proj_idx= new INT[num_vec];
	ASSERT(tmp_proj_idx);
	memset(tmp_proj_idx,0,sizeof(INT)*num_vec);

	grad_proj= new DREAL[num_vec];
	ASSERT(grad_proj);
	memset(grad_proj,0,sizeof(DREAL)*num_vec);

	hinge_point= new DREAL[num_vec+num_feat];
	ASSERT(hinge_point);
	memset(hinge_point,0,sizeof(DREAL)*(num_vec+num_feat));

	hinge_idx= new INT[num_vec+num_feat];
	ASSERT(hinge_idx);
	memset(hinge_idx,0,sizeof(INT)*(num_vec+num_feat));

	active=new BYTE[num_vec];
	ASSERT(active);
	memset(active,0,sizeof(BYTE)*num_vec);

	old_active=new BYTE[num_vec];
	ASSERT(old_active);
	memset(old_active,0,sizeof(BYTE)*num_vec);

	idx_bound=new INT[num_vec];
	ASSERT(idx_bound);
	memset(idx_bound,0,sizeof(INT)*num_vec);

	idx_active=new INT[num_vec];
	ASSERT(idx_active);
	memset(idx_active,0,sizeof(INT)*num_vec);

	beta=new DREAL[num_feat+1+num_feat+num_vec];
	ASSERT(beta);
	memset(beta,0,sizeof(DREAL)*num_feat+1+num_feat+num_vec);

	solver=new CCplex();
}

void CSubGradientLPM::cleanup()
{
	delete[] hinge_idx;
	delete[] hinge_point;
	delete[] grad_proj;
	delete[] proj;
	delete[] tmp_proj;
	delete[] tmp_proj_idx;
	delete[] active;
	delete[] old_active;
	delete[] idx_bound;
	delete[] idx_active;
	delete[] sum_CXy_active;
	delete[] w_pos;
	delete[] w_zero;
	delete[] w_neg;
	delete[] grad_w;
	delete[] beta;

	hinge_idx=NULL;
	proj=NULL;
	active=NULL;
	old_active=NULL;
	idx_bound=NULL;
	idx_active=NULL;
	sum_CXy_active=NULL;
	grad_w=NULL;
	beta=NULL;

	delete solver;
	solver=NULL;
}

bool CSubGradientLPM::train()
{
	lpmtim=0;
	SG_INFO("C=%f epsilon=%f\n", C1, epsilon);
	ASSERT(get_labels());
	ASSERT(get_features());

	INT num_iterations=0;
	INT num_train_labels=get_labels()->get_num_labels();
	INT num_feat=features->get_num_features();
	INT num_vec=features->get_num_vectors();

	ASSERT(num_vec==num_train_labels);

	init(num_vec, num_feat);

	INT num_active=0;
	INT num_bound=0;
	DREAL alpha=0;
	DREAL dir_deriv=0;
	DREAL obj=0;
	delta_active=num_vec;
	last_it_noimprovement=-1;

	work_epsilon=epsilon;
	autoselected_epsilon=work_epsilon;

	compute_projection(num_feat, num_vec);

	double loop_time=0;
	while (!(CSignal::cancel_computations()))
	{
		CTime t;
		delta_active=find_active(num_feat, num_vec, num_active, num_bound);

		update_active(num_feat, num_vec);

#ifdef DEBUG_SUBGRADIENTLPM
		SG_PRINT("==================================================\niteration: %d ", num_iterations);
		obj=compute_objective(num_feat, num_vec);
		SG_PRINT("objective:%.10f alpha: %.10f dir_deriv: %f num_bound: %d num_active: %d work_eps: %10.10f eps: %10.10f auto_eps: %10.10f time:%f\n",
				obj, alpha, dir_deriv, num_bound, num_active, work_epsilon, epsilon, autoselected_epsilon, loop_time);
#else
	  SG_ABS_PROGRESS(work_epsilon, -CMath::log10(work_epsilon), -CMath::log10(0.99999999), -CMath::log10(epsilon), 6);
#endif
		//CMath::display_vector(w, w_dim, "w");
		//SG_PRINT("bias: %f\n", bias);
		//CMath::display_vector(proj, num_vec, "proj");
		//CMath::display_vector(idx_active, num_active, "idx_active");
		//SG_PRINT("num_active: %d\n", num_active);
		//CMath::display_vector(idx_bound, num_bound, "idx_bound");
		//SG_PRINT("num_bound: %d\n", num_bound);
		//CMath::display_vector(sum_CXy_active, num_feat, "sum_CXy_active");
		//SG_PRINT("sum_Cy_active: %f\n", sum_Cy_active);
		//CMath::display_vector(grad_w, num_feat, "grad_w");
		//SG_PRINT("grad_b:%f\n", grad_b);
		
		dir_deriv=compute_min_subgradient(num_feat, num_vec, num_active, num_bound);

		alpha=line_search(num_feat, num_vec);

		if (num_it_noimprovement==10 || num_bound<qpsize_max)
		{
			DREAL norm_grad=CMath::dot(grad_w, grad_w, num_feat) +
				grad_b*grad_b;

			SG_PRINT("CHECKING OPTIMALITY CONDITIONS: "
					"work_epsilon: %10.10f delta_active:%d norm_grad: %10.10f\n", work_epsilon, delta_active, norm_grad);
			if (work_epsilon<=epsilon && delta_active==0 && alpha*norm_grad<1e-12)
				break;
			else
				num_it_noimprovement=0;
		}

		//if (work_epsilon<=epsilon && delta_active==0 && num_it_noimprovement)
		if ((dir_deriv<0 || alpha==0) && (work_epsilon<=epsilon && delta_active==0))
		{
			if (last_it_noimprovement==num_iterations-1)
			{
				SG_PRINT("no improvement...\n");
				num_it_noimprovement++;
			}
			else
				num_it_noimprovement=0;

			last_it_noimprovement=num_iterations;
		}

		CMath::vec1_plus_scalar_times_vec2(w, -alpha, grad_w, num_feat);
		bias-=alpha*grad_b;

		update_projection(alpha, num_vec);
		//compute_projection(num_feat, num_vec);
		CMath::display_vector(w, w_dim, "w");
		SG_PRINT("bias: %f\n", bias);
		CMath::display_vector(proj, num_vec, "proj");

		//if (num_iterations==2)
		//	ASSERT(0);
		t.stop();
		loop_time=t.time_diff_sec();
		num_iterations++;
	}

	SG_INFO("converged after %d iterations\n", num_iterations);

	obj=compute_objective(num_feat, num_vec);
	SG_INFO("objective: %f alpha: %f dir_deriv: %f num_bound: %d num_active: %d sparsity: %f\n",
			obj, alpha, dir_deriv, num_bound, num_active, sparsity/num_iterations);

#ifdef DEBUG_SUBGRADIENTLPM
	CMath::display_vector(w, w_dim, "w");
	SG_PRINT("bias: %f\n", bias);
#endif
	SG_PRINT("solver time:%f s\n", lpmtim);

	cleanup();

	return true;
}
