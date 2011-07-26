/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2007-2009 Soeren Sonnenburg
 * Written (W) 2007-2008 Vojtech Franc
 * Copyright (C) 2007-2009 Fraunhofer Institute FIRST and Max-Planck-Society
 */

#include <shogun/lib/config.h>

#ifdef USE_CPLEX

#include <shogun/mathematics/Math.h>
#include <shogun/lib/Signal.h>
#include <shogun/lib/Time.h>
#include <shogun/machine/LinearMachine.h>
#include <shogun/classifier/SubGradientLPM.h>
#include <shogun/classifier/svm/qpbsvmlib.h>
#include <shogun/features/DotFeatures.h>
#include <shogun/features/Labels.h>

using namespace shogun;

#define DEBUG_SUBGRADIENTLPM

CSubGradientLPM::CSubGradientLPM()
: CLinearClassifier(), C1(1), C2(1), epsilon(1e-5), qpsize(42),
	qpsize_max(2000), use_bias(false), delta_active(0), delta_bound(0)
{
}

CSubGradientLPM::CSubGradientLPM(
	float64_t C, CDotFeatures* traindat, CLabels* trainlab)
: CLinearClassifier(), C1(C), C2(C), epsilon(1e-5), qpsize(42),
	qpsize_max(2000), use_bias(false), delta_active(0), delta_bound(0)
{
	CLinearClassifier::features=traindat;
	CClassifier::labels=trainlab;
}


CSubGradientLPM::~CSubGradientLPM()
{
	cleanup();
}

int32_t CSubGradientLPM::find_active(
	int32_t num_feat, int32_t num_vec, int32_t& num_active, int32_t& num_bound)
{
	//delta_active=0;
	//num_active=0;
	//num_bound=0;

	//for (int32_t i=0; i<num_vec; i++)
	//{
	//	active[i]=0;

	//	//within margin/wrong side
	//	if (proj[i] < 1-work_epsilon)
	//	{
	//		idx_active[num_active++]=i;
	//		active[i]=1;
	//	}

	//	//on margin
	//	if (CMath::abs(proj[i]-1) <= work_epsilon)
	//	{
	//		idx_bound[num_bound++]=i;
	//		active[i]=2;
	//	}

	//	if (active[i]!=old_active[i])
	//		delta_active++;
	//}

	delta_bound=0;
	delta_active=0;
	num_active=0;
	num_bound=0;

	for (int32_t i=0; i<num_vec; i++)
	{
		active[i]=0;

		//within margin/wrong side
		if (proj[i] < 1-autoselected_epsilon)
		{
			idx_active[num_active++]=i;
			active[i]=1;
		}

		//on margin
		if (CMath::abs(proj[i]-1) <= autoselected_epsilon)
		{
			idx_bound[num_bound++]=i;
			active[i]=2;
		}

		if (active[i]!=old_active[i])
			delta_active++;

		if (active[i]==2 && old_active[i]==2)
			delta_bound++;
	}


	if (delta_active==0 && work_epsilon<=epsilon) //we converged
		return 0;
	else if (delta_active==0) //lets decrease work_epsilon
	{
		work_epsilon=CMath::min(work_epsilon/2, autoselected_epsilon);
		work_epsilon=CMath::max(work_epsilon, epsilon);
		num_bound=qpsize;
	}

	delta_bound=0;
	delta_active=0;
	num_active=0;
	num_bound=0;

	for (int32_t i=0; i<num_vec; i++)
	{
		tmp_proj[i]=CMath::abs(proj[i]-1);
		tmp_proj_idx[i]=i;
	}

	CMath::qsort_index(tmp_proj, tmp_proj_idx, num_vec);

	autoselected_epsilon=tmp_proj[CMath::min(qpsize,num_vec)];

#ifdef DEBUG_SUBGRADIENTSVM
	//SG_PRINT("autoseleps: %15.15f\n", autoselected_epsilon);
#endif

	if (autoselected_epsilon>work_epsilon)
		autoselected_epsilon=work_epsilon;

	if (autoselected_epsilon<epsilon)
	{
		autoselected_epsilon=epsilon;

		int32_t i=0;
		while (i < num_vec && tmp_proj[i] <= autoselected_epsilon)
			i++;

		//SG_PRINT("lower bound on epsilon requires %d variables in qp\n", i);

		if (i>=qpsize_max && autoselected_epsilon>epsilon) //qpsize limit
		{
			SG_PRINT("qpsize limit (%d) reached\n", qpsize_max);
			int32_t num_in_qp=i;
			while (--i>=0 && num_in_qp>=qpsize_max)
			{
				if (tmp_proj[i] < autoselected_epsilon)
				{
					autoselected_epsilon=tmp_proj[i];
					num_in_qp--;
				}
			}

			//SG_PRINT("new qpsize will be %d, autoeps:%15.15f\n", num_in_qp, autoselected_epsilon);
		}
	}

	for (int32_t i=0; i<num_vec; i++)
	{
		active[i]=0;

		//within margin/wrong side
		if (proj[i] < 1-autoselected_epsilon)
		{
			idx_active[num_active++]=i;
			active[i]=1;
		}

		//on margin
		if (CMath::abs(proj[i]-1) <= autoselected_epsilon)
		{
			idx_bound[num_bound++]=i;
			active[i]=2;
		}

		if (active[i]!=old_active[i])
			delta_active++;

		if (active[i]==2 && old_active[i]==2)
			delta_bound++;
	}

	pos_idx=0;
	neg_idx=0;
	zero_idx=0;

	for (int32_t i=0; i<num_feat; i++)
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


void CSubGradientLPM::update_active(int32_t num_feat, int32_t num_vec)
{
	for (int32_t i=0; i<num_vec; i++)
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

float64_t CSubGradientLPM::line_search(int32_t num_feat, int32_t num_vec)
{
	int32_t num_hinge=0;
	float64_t alpha=0;
	float64_t sgrad=0;

	float64_t* A=SG_MALLOCX(float64_t, num_feat+num_vec);
	float64_t* B=SG_MALLOCX(float64_t, num_feat+num_vec);
	float64_t* C=SG_MALLOCX(float64_t, num_feat+num_vec);
	float64_t* D=SG_MALLOCX(float64_t, num_feat+num_vec);

	for (int32_t i=0; i<num_feat+num_vec; i++)
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
			float64_t p=get_label(i-num_feat)*(features->dense_dot(i-num_feat, grad_w, num_feat)+grad_b);
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

	//SG_PRINT("sgrad:%f\n", sgrad);
	//CMath::display_vector(A, num_feat+num_vec, "A");
	//CMath::display_vector(B, num_feat+num_vec, "B");
	//CMath::display_vector(C, num_feat+num_vec, "C");
	//CMath::display_vector(D, num_feat+num_vec, "D");
	//CMath::display_vector(hinge_point, num_feat+num_vec, "hinge_point");
	//CMath::display_vector(hinge_idx, num_feat+num_vec, "hinge_idx");
	//ASSERT(0);

	CMath::qsort_index(hinge_point, hinge_idx, num_hinge);
	//CMath::display_vector(hinge_point, num_feat+num_vec, "hinge_point_sorted");


	int32_t i=-1;
	while (i < num_hinge-1 && sgrad < 0)
	{
		i+=1;

		if (A[hinge_idx[i]] > C[hinge_idx[i]])
			sgrad += A[hinge_idx[i]] - C[hinge_idx[i]];
		else
			sgrad += C[hinge_idx[i]] - A[hinge_idx[i]];
	}

	alpha = hinge_point[i];

	SG_FREE(D);
	SG_FREE(C);
	SG_FREE(B);
	SG_FREE(A);

	//SG_PRINT("alpha=%f\n", alpha);
	return alpha;
}

float64_t CSubGradientLPM::compute_min_subgradient(
	int32_t num_feat, int32_t num_vec, int32_t num_active, int32_t num_bound)
{
	float64_t dir_deriv=0;
	solver->init(E_QP);

	if (zero_idx+num_bound > 0)
	{
		//SG_PRINT("num_var:%d (zero:%d, bound:%d) num_feat:%d\n", zero_idx+num_bound, zero_idx,num_bound, num_feat);
		//CMath::display_vector(grad_w, num_feat+1, "grad_w");
		CMath::add(grad_w, 1.0, grad_w, -1.0, sum_CXy_active, num_feat);
		grad_w[num_feat]= -sum_Cy_active;

		grad_b = -sum_Cy_active;

		//CMath::display_vector(sum_CXy_active, num_feat, "sum_CXy_active");
		//SG_PRINT("sum_Cy_active=%10.10f\n", sum_Cy_active);

		//CMath::display_vector(grad_w, num_feat+1, "grad_w");

		solver->setup_subgradientlpm_QP(C1, labels, (CSparseFeatures<float64_t>*) features, idx_bound, num_bound,
				w_zero, zero_idx,
				grad_w, num_feat+1,
				use_bias);

		solver->optimize(beta);
		//CMath::display_vector(beta, num_feat+1, "v");

		//compute dir_deriv here, variable grad_w constains still 'v' and beta
		//contains the future gradient
		dir_deriv = CMath::dot(beta, grad_w, num_feat);
		dir_deriv-=beta[num_feat]*sum_Cy_active;

		for (int32_t i=0; i<num_bound; i++)
		{
			float64_t val= C1*get_label(idx_bound[i])*(features->dense_dot(idx_bound[i], beta, num_feat)+ beta[num_feat]);
			dir_deriv += CMath::max(0.0, val);
		}

		for (int32_t i=0; i<num_feat; i++)
			grad_w[i]=beta[i];

		if (use_bias)
			grad_b=beta[num_feat];

		//for (int32_t i=0; i<zero_idx+num_bound; i++)
		//	beta[i]=beta[i+num_feat+1];

		//CMath::display_vector(beta, zero_idx+num_bound, "beta");
		//SG_PRINT("beta[0]=%10.16f\n", beta[0]);
		//ASSERT(0);

		//for (int32_t i=0; i<zero_idx+num_bound; i++)
		//{
		//	if (i<zero_idx)
		//		grad_w[w_zero[i]]+=beta[w_zero[i]];
		//	else
		//	{
		//		features->add_to_dense_vec(-C1*beta[i]*get_label(idx_bound[i-zero_idx]), idx_bound[i-zero_idx], grad_w, num_feat);
		//		if (use_bias)
		//			grad_b -=  C1 * get_label(idx_bound[i-zero_idx])*beta[i-zero_idx];
		//	}
		//}

		//CMath::display_vector(w_zero, zero_idx, "w_zero");
		//CMath::display_vector(grad_w, num_feat, "grad_w");
		//SG_PRINT("grad_b=%f\n", grad_b);
		//
	}
	else
	{
		CMath::add(grad_w, 1.0, w, -1.0, sum_CXy_active, num_feat);
		grad_b = -sum_Cy_active;

		dir_deriv = CMath::dot(grad_w, grad_w, num_feat)+ grad_b*grad_b;
	}

	solver->cleanup();


	//SG_PRINT("Gradient   : |subgrad_W|^2=%f, |subgrad_b|^2=%f\n",
	//		CMath::dot(grad_w, grad_w, num_feat), grad_b*grad_b);

	return dir_deriv;
}

float64_t CSubGradientLPM::compute_objective(int32_t num_feat, int32_t num_vec)
{
	float64_t result= CMath::sum_abs(w, num_feat);
	
	for (int32_t i=0; i<num_vec; i++)
	{
		if (proj[i]<1.0)
			result += C1 * (1.0-proj[i]);
	}

	return result;
}

void CSubGradientLPM::compute_projection(int32_t num_feat, int32_t num_vec)
{
	for (int32_t i=0; i<num_vec; i++)
		proj[i]=get_label(i)*(features->dense_dot(i, w, num_feat) + bias);
}

void CSubGradientLPM::update_projection(float64_t alpha, int32_t num_vec)
{
	CMath::vec1_plus_scalar_times_vec2(proj,-alpha, grad_proj, num_vec);
}

void CSubGradientLPM::init(int32_t num_vec, int32_t num_feat)
{
	// alloc normal and bias inited with 0
	SG_FREE(w);
	w=SG_MALLOCX(float64_t, num_feat);
	w_dim=num_feat;
	for (int32_t i=0; i<num_feat; i++)
		w[i]=1.0;
	//CMath::random_vector(w, num_feat, -1.0, 1.0);
	bias=0;
	num_it_noimprovement=0;
	grad_b=0;

	w_pos=SG_MALLOCX(int32_t, num_feat);
	memset(w_pos,0,sizeof(int32_t)*num_feat);

	w_zero=SG_MALLOCX(int32_t, num_feat);
	memset(w_zero,0,sizeof(int32_t)*num_feat);

	w_neg=SG_MALLOCX(int32_t, num_feat);
	memset(w_neg,0,sizeof(int32_t)*num_feat);

	grad_w=SG_MALLOCX(float64_t, num_feat+1);
	memset(grad_w,0,sizeof(float64_t)*(num_feat+1));

	sum_CXy_active=SG_MALLOCX(float64_t, num_feat);
	memset(sum_CXy_active,0,sizeof(float64_t)*num_feat);

	sum_Cy_active=0;

	proj=SG_MALLOCX(float64_t, num_vec);
	memset(proj,0,sizeof(float64_t)*num_vec);

	tmp_proj=SG_MALLOCX(float64_t, num_vec);
	memset(proj,0,sizeof(float64_t)*num_vec);

	tmp_proj_idx=SG_MALLOCX(int32_t, num_vec);
	memset(tmp_proj_idx,0,sizeof(int32_t)*num_vec);

	grad_proj=SG_MALLOCX(float64_t, num_vec);
	memset(grad_proj,0,sizeof(float64_t)*num_vec);

	hinge_point=SG_MALLOCX(float64_t, num_vec+num_feat);
	memset(hinge_point,0,sizeof(float64_t)*(num_vec+num_feat));

	hinge_idx=SG_MALLOCX(int32_t, num_vec+num_feat);
	memset(hinge_idx,0,sizeof(int32_t)*(num_vec+num_feat));

	active=SG_MALLOCX(uint8_t, num_vec);
	memset(active,0,sizeof(uint8_t)*num_vec);

	old_active=SG_MALLOCX(uint8_t, num_vec);
	memset(old_active,0,sizeof(uint8_t)*num_vec);

	idx_bound=SG_MALLOCX(int32_t, num_vec);
	memset(idx_bound,0,sizeof(int32_t)*num_vec);

	idx_active=SG_MALLOCX(int32_t, num_vec);
	memset(idx_active,0,sizeof(int32_t)*num_vec);

	beta=SG_MALLOCX(float64_t, num_feat+1+num_feat+num_vec);
	memset(beta,0,sizeof(float64_t)*num_feat+1+num_feat+num_vec);

	solver=new CCplex();
}

void CSubGradientLPM::cleanup()
{
	SG_FREE(hinge_idx);
	SG_FREE(hinge_point);
	SG_FREE(grad_proj);
	SG_FREE(proj);
	SG_FREE(tmp_proj);
	SG_FREE(tmp_proj_idx);
	SG_FREE(active);
	SG_FREE(old_active);
	SG_FREE(idx_bound);
	SG_FREE(idx_active);
	SG_FREE(sum_CXy_active);
	SG_FREE(w_pos);
	SG_FREE(w_zero);
	SG_FREE(w_neg);
	SG_FREE(grad_w);
	SG_FREE(beta);

	hinge_idx=NULL;
	hinge_point=NULL;
	grad_proj=NULL;
	proj=NULL;
	tmp_proj=NULL;
	tmp_proj_idx=NULL;
	active=NULL;
	old_active=NULL;
	idx_bound=NULL;
	idx_active=NULL;
	sum_CXy_active=NULL;
	w_pos=NULL;
	w_zero=NULL;
	w_neg=NULL;
	grad_w=NULL;
	beta=NULL;

	delete solver;
	solver=NULL;
}

bool CSubGradientLPM::train(CFeatures* data)
{
	lpmtim=0;
	SG_INFO("C=%f epsilon=%f\n", C1, epsilon);
	ASSERT(labels);
	if (data)
	{
		if (!data->has_property(FP_DOT))
			SG_ERROR("Specified features are not of type CDotFeatures\n");
		set_features((CDotFeatures*) data);
	}
	ASSERT(features);

	int32_t num_iterations=0;
	int32_t num_train_labels=labels->get_num_labels();
	int32_t num_feat=features->get_dim_feature_space();
	int32_t num_vec=features->get_num_vectors();

	ASSERT(num_vec==num_train_labels);

	init(num_vec, num_feat);

	int32_t num_active=0;
	int32_t num_bound=0;
	float64_t alpha=0;
	float64_t dir_deriv=0;
	float64_t obj=0;
	delta_active=num_vec;
	last_it_noimprovement=-1;

	work_epsilon=0.99;
	autoselected_epsilon=work_epsilon;

	compute_projection(num_feat, num_vec);

	CTime time;
	float64_t loop_time=0;
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
			float64_t norm_grad=CMath::dot(grad_w, grad_w, num_feat) +
				grad_b*grad_b;

			SG_PRINT("CHECKING OPTIMALITY CONDITIONS: "
					"work_epsilon: %10.10f delta_active:%d alpha: %10.10f norm_grad: %10.10f a*norm_grad:%10.16f\n",
					work_epsilon, delta_active, alpha, norm_grad, CMath::abs(alpha*norm_grad));

			if (work_epsilon<=epsilon && delta_active==0 && CMath::abs(alpha*norm_grad)<1e-6)
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

		t.stop();
		loop_time=t.time_diff_sec();
		num_iterations++;

		if (get_max_train_time()>0 && time.cur_time_diff()>get_max_train_time())
			break;
	}

	SG_INFO("converged after %d iterations\n", num_iterations);

	obj=compute_objective(num_feat, num_vec);
	SG_INFO("objective: %f alpha: %f dir_deriv: %f num_bound: %d num_active: %d\n",
			obj, alpha, dir_deriv, num_bound, num_active);

#ifdef DEBUG_SUBGRADIENTLPM
	CMath::display_vector(w, w_dim, "w");
	SG_PRINT("bias: %f\n", bias);
#endif
	SG_PRINT("solver time:%f s\n", lpmtim);

	cleanup();

	return true;
}
#endif //USE_CPLEX
