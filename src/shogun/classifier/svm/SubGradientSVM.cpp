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
#include <shogun/lib/Mathematics.h>
#include <shogun/lib/Signal.h>
#include <shogun/lib/Time.h>
#include <shogun/machine/LinearMachine.h>
#include <shogun/classifier/svm/SubGradientSVM.h>
#include <shogun/classifier/svm/QPBSVMLib.h>
#include <shogun/features/DotFeatures.h>
#include <shogun/features/Labels.h>

#undef DEBUG_SUBGRADIENTSVM

using namespace shogun;

CSubGradientSVM::CSubGradientSVM()
: CLinearMachine(), C1(1), C2(1), epsilon(1e-5), qpsize(42),
	qpsize_max(2000), use_bias(false), delta_active(0), delta_bound(0)
{
}

CSubGradientSVM::CSubGradientSVM(
	float64_t C, CDotFeatures* traindat, CLabels* trainlab)
: CLinearMachine(), C1(C), C2(C), epsilon(1e-5), qpsize(42),
	qpsize_max(2000), use_bias(false), delta_active(0), delta_bound(0)
{
	set_features(traindat);
	set_labels(trainlab);
}


CSubGradientSVM::~CSubGradientSVM()
{
}

/*
int32_t CSubGradientSVM::find_active(int32_t num_feat, int32_t num_vec, int32_t& num_active, int32_t& num_bound)
{
	int32_t delta_active=0;
	num_active=0;
	num_bound=0;

	for (int32_t i=0; i<num_vec; i++)
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

	return delta_active;
}
*/

int32_t CSubGradientSVM::find_active(
	int32_t num_feat, int32_t num_vec, int32_t& num_active, int32_t& num_bound)
{
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

	autoselected_epsilon=tmp_proj[CMath::min(qpsize,num_vec-1)];

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
			SG_INFO("qpsize limit (%d) reached\n", qpsize_max);
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

	//SG_PRINT("delta_bound: %d of %d (%02.2f)\n", delta_bound, num_bound, 100.0*delta_bound/num_bound);
	return delta_active;
}


void CSubGradientSVM::update_active(int32_t num_feat, int32_t num_vec)
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

float64_t CSubGradientSVM::line_search(int32_t num_feat, int32_t num_vec)
{
	float64_t sum_B = 0;
	float64_t A_zero = 0.5*CMath::dot(grad_w, grad_w, num_feat);
	float64_t B_zero = -CMath::dot(w, grad_w, num_feat);

	int32_t num_hinge=0;

	for (int32_t i=0; i<num_vec; i++)
	{
		float64_t p=get_label(i)*(features->dense_dot(i, grad_w, num_feat)+grad_b);
		grad_proj[i]=p;
		if (p!=0)
		{
			hinge_point[num_hinge]=(proj[i]-1)/p;
			hinge_idx[num_hinge]=i;
			num_hinge++;

			if (p<0)
				sum_B+=p;
		}
	}
	sum_B*=C1;

	CMath::qsort_index(hinge_point, hinge_idx, num_hinge);


	float64_t alpha = hinge_point[0];
	float64_t grad_val = 2*A_zero*alpha + B_zero + sum_B;

	//CMath::display_vector(grad_w, num_feat, "grad_w");
	//CMath::display_vector(grad_proj, num_vec, "grad_proj");
	//CMath::display_vector(hinge_point, num_vec, "hinge_point");
	//SG_PRINT("A_zero=%f\n", A_zero);
	//SG_PRINT("B_zero=%f\n", B_zero);
	//SG_PRINT("sum_B=%f\n", sum_B);
	//SG_PRINT("alpha=%f\n", alpha);
	//SG_PRINT("grad_val=%f\n", grad_val);

	float64_t old_grad_val = grad_val;
	float64_t old_alpha = alpha;

	for (int32_t i=1; i < num_hinge && grad_val < 0; i++)
	{
		alpha = hinge_point[i];
		grad_val = 2*A_zero*alpha + B_zero + sum_B;

		if (grad_val > 0)
		{
			ASSERT(old_grad_val-grad_val != 0);
			float64_t gamma = -grad_val/(old_grad_val-grad_val);
			alpha = old_alpha*gamma + (1-gamma)*alpha;
		}
		else
		{
			old_grad_val = grad_val;
			old_alpha = alpha;

			sum_B = sum_B + CMath::abs(C1*grad_proj[hinge_idx[i]]);
			grad_val = 2*A_zero*alpha + B_zero + sum_B;
		}
	}

	return alpha;
}

float64_t CSubGradientSVM::compute_min_subgradient(
	int32_t num_feat, int32_t num_vec, int32_t num_active, int32_t num_bound)
{
	float64_t dir_deriv=0;

	if (num_bound > 0)
	{

		CTime t2;
		CMath::add(v, 1.0, w, -1.0, sum_CXy_active, num_feat);

		if (num_bound>=qpsize_max && num_it_noimprovement!=10) // if qp gets to large, lets just choose a random beta
		{
			//SG_PRINT("qpsize too large  (%d>=%d) choosing random subgradient/beta\n", num_bound, qpsize_max);
			for (int32_t i=0; i<num_bound; i++)
				beta[i]=CMath::random(0.0,1.0);
		}
		else
		{
			memset(beta, 0, sizeof(float64_t)*num_bound);

			float64_t bias_const=0;

			if (use_bias)
				bias_const=1;

			for (int32_t i=0; i<num_bound; i++)
			{
				for (int32_t j=i; j<num_bound; j++)
				{
					Z[i*num_bound+j]= 2.0*C1*C1*get_label(idx_bound[i])*get_label(idx_bound[j])* 
						(features->dot(idx_bound[i], features, idx_bound[j]) + bias_const);

					Z[j*num_bound+i]=Z[i*num_bound+j];

				}

				Zv[i]=-2.0*C1*get_label(idx_bound[i])* 
					(features->dense_dot(idx_bound[i], v, num_feat)-sum_Cy_active);
			}

			//CMath::display_matrix(Z, num_bound, num_bound, "Z");
			//CMath::display_vector(Zv, num_bound, "Zv");
			t2.stop();
#ifdef DEBUG_SUBGRADIENTSVM
			t2.time_diff_sec(true);
#endif

			CTime t;
			CQPBSVMLib solver(Z,num_bound, Zv,num_bound, 1.0);
			//solver.set_solver(QPB_SOLVER_GRADDESC);
			//solver.set_solver(QPB_SOLVER_GS);
#ifdef USE_CPLEX
			solver.set_solver(QPB_SOLVER_CPLEX);
#else
			solver.set_solver(QPB_SOLVER_SCAS);
#endif

			solver.solve_qp(beta, num_bound);

			t.stop();
#ifdef DEBUG_SUBGRADIENTSVM
			tim+=t.time_diff_sec(true);
#else
			tim+=t.time_diff_sec(false);
#endif

			//CMath::display_vector(beta, num_bound, "beta gs");
			//solver.set_solver(QPB_SOLVER_CPLEX);
			//solver.solve_qp(beta, num_bound);
			//CMath::display_vector(beta, num_bound, "beta cplex");

			//CMath::display_vector(grad_w, num_feat, "grad_w");
			//SG_PRINT("grad_b:%f\n", grad_b);
		}

		CMath::add(grad_w, 1.0, w, -1.0, sum_CXy_active, num_feat);
		grad_b = -sum_Cy_active;

		for (int32_t i=0; i<num_bound; i++)
		{
			features->add_to_dense_vec(-C1*beta[i]*get_label(idx_bound[i]), idx_bound[i], grad_w, num_feat);
			if (use_bias)
				grad_b -=  C1 * get_label(idx_bound[i])*beta[i];
		}

		dir_deriv = CMath::dot(grad_w, v, num_feat) - grad_b*sum_Cy_active;
		for (int32_t i=0; i<num_bound; i++)
		{
			float64_t val= C1*get_label(idx_bound[i])*(features->dense_dot(idx_bound[i], grad_w, num_feat)+grad_b);
			dir_deriv += CMath::max(0.0, val);
		}
	}
	else
	{
		CMath::add(grad_w, 1.0, w, -1.0, sum_CXy_active, num_feat);
		grad_b = -sum_Cy_active;

		dir_deriv = CMath::dot(grad_w, grad_w, num_feat)+ grad_b*grad_b;
	}

	return dir_deriv;
}

float64_t CSubGradientSVM::compute_objective(int32_t num_feat, int32_t num_vec)
{
	float64_t result= 0.5 * CMath::dot(w,w, num_feat);
	
	for (int32_t i=0; i<num_vec; i++)
	{
		if (proj[i]<1.0)
			result += C1 * (1.0-proj[i]);
	}

	return result;
}

void CSubGradientSVM::compute_projection(int32_t num_feat, int32_t num_vec)
{
	for (int32_t i=0; i<num_vec; i++)
		proj[i]=get_label(i)*(features->dense_dot(i, w, num_feat)+bias);
}

void CSubGradientSVM::update_projection(float64_t alpha, int32_t num_vec)
{
	CMath::vec1_plus_scalar_times_vec2(proj,-alpha, grad_proj, num_vec);
}

void CSubGradientSVM::init(int32_t num_vec, int32_t num_feat)
{
	// alloc normal and bias inited with 0
	delete[] w;
	w=new float64_t[num_feat];
	w_dim=num_feat;
	memset(w,0,sizeof(float64_t)*num_feat);
	//CMath::random_vector(w, num_feat, -1.0, 1.0);
	bias=0;
	num_it_noimprovement=0;
	grad_b=0;
	qpsize_limit=5000;

	grad_w=new float64_t[num_feat];
	memset(grad_w,0,sizeof(float64_t)*num_feat);

	sum_CXy_active=new float64_t[num_feat];
	memset(sum_CXy_active,0,sizeof(float64_t)*num_feat);

	v=new float64_t[num_feat];
	memset(v,0,sizeof(float64_t)*num_feat);

	old_v=new float64_t[num_feat];
	memset(old_v,0,sizeof(float64_t)*num_feat);

	sum_Cy_active=0;

	proj= new float64_t[num_vec];
	memset(proj,0,sizeof(float64_t)*num_vec);

	tmp_proj=new float64_t[num_vec];
	memset(proj,0,sizeof(float64_t)*num_vec);

	tmp_proj_idx= new int32_t[num_vec];
	memset(tmp_proj_idx,0,sizeof(int32_t)*num_vec);

	grad_proj= new float64_t[num_vec];
	memset(grad_proj,0,sizeof(float64_t)*num_vec);

	hinge_point= new float64_t[num_vec];
	memset(hinge_point,0,sizeof(float64_t)*num_vec);

	hinge_idx= new int32_t[num_vec];
	memset(hinge_idx,0,sizeof(int32_t)*num_vec);

	active=new uint8_t[num_vec];
	memset(active,0,sizeof(uint8_t)*num_vec);

	old_active=new uint8_t[num_vec];
	memset(old_active,0,sizeof(uint8_t)*num_vec);

	idx_bound=new int32_t[num_vec];
	memset(idx_bound,0,sizeof(int32_t)*num_vec);

	idx_active=new int32_t[num_vec];
	memset(idx_active,0,sizeof(int32_t)*num_vec);

	Z=new float64_t[qpsize_limit*qpsize_limit];
	memset(Z,0,sizeof(float64_t)*qpsize_limit*qpsize_limit);

	Zv=new float64_t[qpsize_limit];
	memset(Zv,0,sizeof(float64_t)*qpsize_limit);

	beta=new float64_t[qpsize_limit];
	memset(beta,0,sizeof(float64_t)*qpsize_limit);

	old_Z=new float64_t[qpsize_limit*qpsize_limit];
	memset(old_Z,0,sizeof(float64_t)*qpsize_limit*qpsize_limit);

	old_Zv=new float64_t[qpsize_limit];
	memset(old_Zv,0,sizeof(float64_t)*qpsize_limit);

	old_beta=new float64_t[qpsize_limit];
	memset(old_beta,0,sizeof(float64_t)*qpsize_limit);

}

void CSubGradientSVM::cleanup()
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
	delete[] grad_w;
	delete[] v;
	delete[] Z;
	delete[] Zv;
	delete[] beta;
	delete[] old_v;
	delete[] old_Z;
	delete[] old_Zv;
	delete[] old_beta;

	hinge_idx=NULL;
	proj=NULL;
	active=NULL;
	old_active=NULL;
	idx_bound=NULL;
	idx_active=NULL;
	sum_CXy_active=NULL;
	grad_w=NULL;
	v=NULL;
	Z=NULL;
	Zv=NULL;
	beta=NULL;
}

bool CSubGradientSVM::train(CFeatures* data)
{
	tim=0;
	SG_INFO("C=%f epsilon=%f\n", C1, epsilon);
	ASSERT(labels);

	if (data)
	{
		if (!data->has_property(FP_DOT))
			SG_ERROR("Specified features are not of type CDotFeatures\n");
		set_features((CDotFeatures*) data);
	}
	ASSERT(get_features());

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

#ifdef DEBUG_SUBGRADIENTSVM
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

#ifdef DEBUG_SUBGRADIENTSVM
			SG_PRINT("CHECKING OPTIMALITY CONDITIONS: "
					"work_epsilon: %10.10f delta_active:%d alpha: %10.10f norm_grad: %10.10f a*norm_grad:%10.16f\n",
					work_epsilon, delta_active, alpha, norm_grad, CMath::abs(alpha*norm_grad));
#else
			SG_ABS_PROGRESS(work_epsilon, -CMath::log10(work_epsilon), -CMath::log10(0.99999999), -CMath::log10(epsilon), 6);
#endif

			if (work_epsilon<=epsilon && delta_active==0 && CMath::abs(alpha*norm_grad)<1e-6)
				break;
			else
				num_it_noimprovement=0;
		}

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
		//CMath::display_vector(w, w_dim, "w");
		//SG_PRINT("bias: %f\n", bias);
		//CMath::display_vector(proj, num_vec, "proj");

		t.stop();
		loop_time=t.time_diff_sec();
		num_iterations++;

		if (get_max_train_time()>0 && time.cur_time_diff()>get_max_train_time())
			break;
	}
	SG_DONE();

	SG_INFO("converged after %d iterations\n", num_iterations);

	obj=compute_objective(num_feat, num_vec);
	SG_INFO("objective: %f alpha: %f dir_deriv: %f num_bound: %d num_active: %d\n",
			obj, alpha, dir_deriv, num_bound, num_active);

#ifdef DEBUG_SUBGRADIENTSVM
	CMath::display_vector(w, w_dim, "w");
	SG_PRINT("bias: %f\n", bias);
#endif
	SG_INFO("solver time:%f s\n", tim);

	cleanup();

	return true;
}
