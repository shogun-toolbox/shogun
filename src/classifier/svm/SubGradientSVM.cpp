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
#include "classifier/svm/SubGradientSVM.h"
#include "classifier/svm/qpbsvmlib.h"
#include "features/SparseFeatures.h"
#include "features/Labels.h"

#define DEBUG_SUBGRADIENTSVM

double tim;

CSubGradientSVM::CSubGradientSVM() : CSparseLinearClassifier(), C1(1), C2(1), epsilon(1e-5), qpsize(100)
{
}

CSubGradientSVM::CSubGradientSVM(DREAL C, CSparseFeatures<DREAL>* traindat, CLabels* trainlab) 
	: CSparseLinearClassifier(), C1(C), C2(C), epsilon(1e-5), qpsize(100)
{
	CSparseLinearClassifier::features=traindat;
	CClassifier::labels=trainlab;
}


CSubGradientSVM::~CSubGradientSVM()
{
}

/*
INT CSubGradientSVM::find_active(INT num_feat, INT num_vec, INT& num_active, INT& num_bound)
{
	INT delta_active=0;
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

	return delta_active;
}
*/

INT CSubGradientSVM::find_active(INT num_feat, INT num_vec, INT& num_active, INT& num_bound)
{
	INT delta_active=0;
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

	if (delta_active!=0 && num_bound>qpsize)
	{
		delta_active=0;
		num_active=0;
		num_bound=0;

		for (INT i=0; i<num_vec; i++)
		{
			tmp_proj[i]=CMath::abs(proj[i]-1);
			tmp_proj_idx[i]=i;
		}

		CMath::qsort(tmp_proj, tmp_proj_idx, num_vec);

		autoselected_epsilon=tmp_proj[CMath::min(qpsize,num_vec)];
		//SG_PRINT("autoseleps: %10.10f\n", autoselected_epsilon);

		if (autoselected_epsilon>work_epsilon)
			autoselected_epsilon=work_epsilon;
		if (autoselected_epsilon<epsilon)
			autoselected_epsilon=epsilon;


		for (INT i=0; i<num_vec; i++)
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
		}
	}

	return delta_active;
}


void CSubGradientSVM::update_active(INT num_feat, INT num_vec)

{
	for (INT i=0; i<num_vec; i++)
	{
		if (active[i]==1 && old_active[i]!=1)
		{
			features->add_to_dense_vec(C1*get_label(i), i, sum_CXy_active, num_feat);
			sum_Cy_active+=C1*get_label(i);
		}
		else if (old_active[i]==1 && active[i]!=1)
		{
			features->add_to_dense_vec(-C1*get_label(i), i, sum_CXy_active, num_feat);
			sum_Cy_active-=C1*get_label(i);
		}
	}

	CMath::swap(active,old_active);
}

DREAL CSubGradientSVM::line_search(INT num_feat, INT num_vec)
{
   DREAL sum_B = 0;
   DREAL A0 = 0.5*CMath::dot(grad_w, grad_w, num_feat);
   DREAL B0 = -CMath::dot(w, grad_w, num_feat);
   
	for (INT i=0; i<num_vec; i++)
	{
		DREAL p=get_label(i)*features->dense_dot(1.0, i, grad_w, num_feat, grad_b);
		grad_proj[i]=p;
		if (p==0)
			p=1e-10;
		hinge_point[i]=(proj[i]-1)/p;
		hinge_idx[i]=i;
		if (p<0)
			sum_B+=p;
	}
	sum_B*=C1;
   
   CMath::qsort(hinge_point, hinge_idx, num_vec);


   DREAL alpha = hinge_point[0];
   DREAL grad_val = 2*A0*alpha + B0 + sum_B;

   //CMath::display_vector(grad_w, num_feat, "grad_w");
   //CMath::display_vector(grad_proj, num_vec, "grad_proj");
   //CMath::display_vector(hinge_point, num_vec, "hinge_point");
   //SG_PRINT("A0=%f\n", A0);
   //SG_PRINT("B0=%f\n", B0);
   //SG_PRINT("sum_B=%f\n", sum_B);
   //SG_PRINT("alpha=%f\n", alpha);
   //SG_PRINT("grad_val=%f\n", grad_val);

   for (INT i=0; i < num_vec && grad_val < 0; i++)
   {
	   DREAL old_grad_val = grad_val;
	   DREAL old_alpha = alpha;

	   alpha = hinge_point[i];
	   grad_val = 2*A0*alpha + B0 + sum_B;

	   if (grad_val > 0)
	   {
		   ASSERT(old_grad_val-grad_val != 0);
		   DREAL beta = -grad_val/(old_grad_val-grad_val);
		   alpha = old_alpha*beta + (1-beta)*alpha;
	   }
	   else
	   {
		   old_grad_val = grad_val;
		   old_alpha = alpha;

		   sum_B = sum_B + CMath::abs(C1*grad_proj[hinge_idx[i]]);
		   grad_val = 2*A0*alpha + B0 + sum_B;
	   }
   }

   return alpha;
}

DREAL CSubGradientSVM::compute_min_subgradient(INT num_feat, INT num_vec, INT num_active, INT num_bound)
{
	DREAL dir_deriv=0;

	if (num_bound > 0)
	{
		DREAL* v=new DREAL[num_feat];
		DREAL* Z=new DREAL[num_bound*num_bound];
		DREAL* Zv=new DREAL[num_bound];
		DREAL* beta=new DREAL[num_bound];

		ASSERT(v);
		ASSERT(Z);
		ASSERT(Zv);
		ASSERT(beta);

		memset(beta, 0, sizeof(DREAL)*num_bound);

		CMath::add(v, 1.0, w, -1.0, sum_CXy_active, num_feat);

		for (INT i=0; i<num_bound; i++)
		{
			for (INT j=0; j<num_bound; j++)
			{
				INT alen=0;
				INT blen=0;
				bool afree=false;
				bool bfree=false;

				TSparseEntry<DREAL>* avec=features->get_sparse_feature_vector(idx_bound[i], alen, afree);
				TSparseEntry<DREAL>* bvec=features->get_sparse_feature_vector(idx_bound[j], blen, bfree);

				Z[i*num_bound+j]= 2.0*C1*C1*get_label(idx_bound[i])*get_label(idx_bound[j])* 
					(features->sparse_dot(1.0, avec,alen, bvec,blen) + 1);

				features->free_feature_vector(avec, idx_bound[i], afree);
				features->free_feature_vector(bvec, idx_bound[j], bfree);
			}

			Zv[i]=-2.0*C1*get_label(idx_bound[i])* 
				features->dense_dot(1.0, idx_bound[i], v, num_feat, -sum_Cy_active);
		}

		//SG_PRINT("num_bound:%d\n", num_bound);
		//CMath::display_matrix(Z, num_bound, num_bound, "Z");
		//CMath::display_vector(Zv, num_bound, "Zv");

		//SG_PRINT("solver start\n");
		CTime t;
		CQPBSVMLib solver(Z,num_bound, Zv,num_bound, 1.0);
		//solver.set_solver(QPB_SOLVER_SCAMV);
//#ifdef USE_CPLEX
		//solver.set_solver(QPB_SOLVER_CPLEX);
//#else
		//solver.set_solver(QPB_SOLVER_SCA);
//#endif
		//solver.set_solver(QPB_SOLVER_SCAS);
		//solver.set_solver(QPB_SOLVER_PRLOQO);
		//
		SG_PRINT("CPLEX\n");
		solver.set_solver(QPB_SOLVER_CPLEX);
		solver.solve_qp(beta, num_bound);
		CMath::display_vector(beta, num_bound);

		SG_PRINT("SCA\n");
		solver.set_solver(QPB_SOLVER_SCA);
		solver.solve_qp(beta, num_bound);
		CMath::display_vector(beta, num_bound);

		SG_ERROR("stop");

		t.stop();
		tim+=t.time_diff_sec(true);
		//SG_PRINT("solver stop\n");

		//SG_PRINT("after solveer foo\n");
		
		//CMath::display_vector(beta, num_bound, "beta");

		//CMath::display_vector(grad_w, num_feat, "grad_w");
		//SG_PRINT("grad_b:%f\n", grad_b);

		CMath::add(grad_w, 1.0, w, -1.0, sum_CXy_active, num_feat);
		grad_b = -sum_Cy_active;

		for (INT i=0; i<num_bound; i++)
		{
			features->add_to_dense_vec(-C1*beta[i]*get_label(idx_bound[i]), idx_bound[i], grad_w, num_feat);
			grad_b -=  C1 * get_label(idx_bound[i])*beta[i];
		}

#ifdef DEBUG_SUBGRADIENTSVM
		dir_deriv = CMath::dot(grad_w, v, num_feat) - grad_b*sum_Cy_active;
		for (INT i=0; i<num_bound; i++)
		{
			DREAL val= features->dense_dot(get_label(idx_bound[i]), idx_bound[i], grad_w, num_feat, grad_b);
			dir_deriv += CMath::max(0.0, val);
		}
		//CMath::display_vector(grad_w, num_feat, "grad_w");
		//SG_PRINT("grad_b:%f\n", grad_b);
		//ASSERT(0);
#endif

		delete[] v;
		delete[] Z;
		delete[] Zv;
		delete[] beta;
	}
	else
	{
		CMath::add(grad_w, 1.0, w, -1.0, sum_CXy_active, num_feat);
		grad_b = -sum_Cy_active;

#ifdef DEBUG_SUBGRADIENTSVM
		dir_deriv = CMath::dot(grad_w, grad_w, num_feat)+ grad_b*grad_b;
#endif
	}

	return dir_deriv;
}

DREAL CSubGradientSVM::compute_objective(INT num_feat, INT num_vec)
{
	DREAL result= 0.5 * CMath::dot(w,w, num_feat);
	
	for (INT i=0; i<num_vec; i++)
	{
		if (proj[i]<1.0)
			result += C1 * (1.0-proj[i]);
	}

	return result;
}

void CSubGradientSVM::compute_projection(INT num_feat, INT num_vec)
{
	for (INT i=0; i<num_vec; i++)
		proj[i]=get_label(i)*features->dense_dot(1.0, i, w, num_feat, bias);
}

void CSubGradientSVM::update_projection(DREAL alpha, INT num_vec)
{
	CMath::vec1_plus_scalar_times_vec2(proj,-alpha, grad_proj, num_vec);
}

void CSubGradientSVM::init(INT num_vec, INT num_feat)
{
	// alloc normal and bias inited with 0
	delete[] w;
	w=new DREAL[num_feat];
	ASSERT(w);
	memset(w,0,sizeof(DREAL)*num_feat);
	bias=0;
	grad_b=0;
	set_w(w, num_feat);

	grad_w=new DREAL[num_feat];
	ASSERT(grad_w);
	memset(grad_w,0,sizeof(DREAL)*num_feat);

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

	hinge_point= new DREAL[num_vec];
	ASSERT(hinge_point);
	memset(hinge_point,0,sizeof(DREAL)*num_vec);

	hinge_idx= new INT[num_vec];
	ASSERT(hinge_idx);
	memset(hinge_idx,0,sizeof(INT)*num_vec);

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

	hinge_idx=NULL;
	proj=NULL;
	active=NULL;
	old_active=NULL;
	idx_bound=NULL;
	idx_active=NULL;
	sum_CXy_active=NULL;
	grad_w=NULL;
}

bool CSubGradientSVM::train()
{
	tim=0;
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
	INT delta_active=num_vec;

	work_epsilon=0.999;
	autoselected_epsilon=work_epsilon;

	compute_projection(num_feat, num_vec);

	while (delta_active>0 && !(CSignal::cancel_computations()))
	{
		while ((delta_active=find_active(num_feat, num_vec, num_active, num_bound))==0)
		{
			if (work_epsilon<=epsilon)
				break;
			else
			{
				work_epsilon/=2;

				if (work_epsilon<epsilon)
					work_epsilon=epsilon;
			}
		}

		if (work_epsilon<=epsilon && delta_active==0)
			break;

		//if (num_bound>100)
		//	SG_WARNING("number of variables at bound is > 100, convergence will be slow\n");

		update_active(num_feat, num_vec);

#ifdef DEBUG_SUBGRADIENTSVM
		SG_PRINT("==================================================\niteration: %d ", num_iterations);
		SG_PRINT("alpha: %f dir_deriv: %f num_bound: %d num_active: %d work_eps: %10.10f eps: %10.10f auto_eps: %10.10f\n",
				alpha, dir_deriv, num_bound, num_active, work_epsilon, epsilon, autoselected_epsilon);
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
		//obj=compute_objective(num_feat, num_vec);
		//SG_PRINT("objective: %f alpha: %f dir_deriv: %f num_bound: %d num_active: %d\n",
		//		obj, alpha, dir_deriv, num_bound, num_active);
		//CMath::display_vector(grad_w, num_feat, "grad_w");
		//SG_PRINT("grad_b:%f\n", grad_b);
		
		dir_deriv=compute_min_subgradient(num_feat, num_vec, num_active, num_bound);
		alpha=line_search(num_feat, num_vec);

		CMath::vec1_plus_scalar_times_vec2(w, -alpha, grad_w, num_feat);
		bias-=alpha*grad_b;

		update_projection(alpha, num_vec);
		//compute_projection(num_feat, num_vec);
		//CMath::display_vector(w, w_dim, "w");
		//SG_PRINT("bias: %f\n", bias);
		//CMath::display_vector(proj, num_vec, "proj");

		num_iterations++;
	}

	SG_INFO("converged after %d iterations\n", num_iterations);
	//obj=compute_objective(num_feat, num_vec);

	obj= 0.5 * CMath::dot(w,w, num_feat);
	
	for (INT i=0; i<num_vec; i++)
	{
		DREAL v=classify_example(i);
		if (v<1.0)
			obj += C1 * (1.0-v);
	}

	SG_INFO("objective: %f alpha: %f dir_deriv: %f num_bound: %d num_active: %d\n",
			obj, alpha, dir_deriv, num_bound, num_active);

#ifdef DEBUG_SUBGRADIENTSVM
	CMath::display_vector(w, w_dim, "w");
	SG_PRINT("bias: %f\n", bias);
#endif
	SG_PRINT("solver time:%f s\n", tim);

	cleanup();

	return true;
}
