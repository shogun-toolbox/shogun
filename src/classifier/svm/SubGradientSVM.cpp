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

#include "classifier/svm/SubGradientSVM.h"
#include "features/Labels.h"
#include "lib/Mathematics.h"
#include "classifier/SparseLinearClassifier.h"
#include "classifier/svm/qpbsvmlib.h"
#include "features/SparseFeatures.h"
#include "features/Labels.h"

CSubGradientSVM::CSubGradientSVM() : CSparseLinearClassifier(), C1(1), C2(1), epsilon(1e-5)
{
}

CSubGradientSVM::CSubGradientSVM(DREAL C, CSparseFeatures<DREAL>* traindat, CLabels* trainlab) 
	: CSparseLinearClassifier(), C1(C), C2(C), epsilon(1e-5)
{
	CSparseLinearClassifier::features=traindat;
	CClassifier::labels=trainlab;
}


CSubGradientSVM::~CSubGradientSVM()
{
}

INT CSubGradientSVM::find_active(INT num_feat, INT num_vec, INT& num_active, INT& num_bound)
{
	INT delta_active=0;
	num_active=0;
	num_bound=0;

	for (INT i=0; i<num_vec; i++)
	{
		active[i]=0;

		//within margin/wrong side
		if (proj[i] < 1-epsilon)
		{
			idx_active[num_active++]=i;
			active[i]=1;
		}

		//on margin
		if (CMath::abs(proj[i]-1) <= epsilon)
		{
			idx_bound[num_bound++]=i;
			active[i]=2;
		}

		if (active[i]!=old_active[i])
			delta_active++;

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

	return delta_active;
}

DREAL CSubGradientSVM::line_search()
{
	/*
   A0 = 0.5*norm(grad_W)^2;
   B0 = -W'*grad_W;
   C0 = 0.5*norm(W)^2;      
   
   grad_proj = data.y.*(data.X'*grad_W+grad_b);
   B = reg_C*grad_proj;
   C = reg_C*(1-proj);
   
   hinge_point = -C./B;
   [dummy,idx] = sort(hinge_point);

   sum_B = sum(B(find(B < 0)));
   alpha = hinge_point(idx(1));
   grad_val = 2*A0*alpha + B0 + sum_B;

   i = 0;
   while i < nData && grad_val < 0,
   
     i = i + 1;
   
     old_grad_val = grad_val;
     old_alpha = alpha;

     alpha = hinge_point(idx(i));
     grad_val = 2*A0*alpha + B0 + sum_B;
     
     if grad_val > 0,
       beta = -grad_val/(old_grad_val-grad_val);
       alpha = old_alpha*beta + (1-beta)*alpha;
     else
       
       old_grad_val = grad_val;
       old_alpha = alpha;
     
       sum_B = sum_B + abs(B(idx(i)));
       grad_val = 2*A0*alpha + B0 + sum_B;
     
     end
   end
   */
	return 0.001;
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

				Z[i+num_bound+j]= 2.0*C1*C1*get_label(idx_bound[i])*get_label(idx_bound[j])* 
					(features->sparse_dot(1.0, avec,alen, bvec,blen) + 1);

				features->free_feature_vector(avec, idx_bound[i], afree);
				features->free_feature_vector(bvec, idx_bound[j], bfree);
			}

			Zv[i]=2.0*get_label(idx_bound[i])* 
				features->dense_dot(1.0, idx_bound[i], v, num_feat, -sum_Cy_active);
		}

		CQPBSVMLib solver(Z,num_bound, Zv,num_bound, 1.0);
		solver.solve_qp(beta, num_bound);
		
		CMath::add(grad_w, 1.0, w, -1.0, sum_CXy_active, num_feat);
		grad_b = -sum_Cy_active;

		for (INT i=0; i<num_bound; i++)
		{
			features->add_to_dense_vec(-C1*beta[i]*get_label(idx_bound[i]), idx_bound[i], grad_w, num_feat);
			grad_b +=  C1 * get_label(idx_bound[i])*beta[i];
		}

		dir_deriv = CMath::dot(grad_w, v, num_feat) - grad_b*sum_Cy_active;
		for (INT i=0; i<num_bound; i++)
		{
			DREAL val= features->dense_dot(get_label(idx_bound[i]), idx_bound[i], grad_w, num_feat, grad_b);
			dir_deriv += CMath::max(0.0, val);
		}

		delete[] v;
		delete[] Z;
		delete[] Zv;
		delete[] beta;
	}
	else
	{
		CMath::add(grad_w, 1.0, w, -1.0, sum_CXy_active, num_feat);
		grad_b = -sum_Cy_active;

		dir_deriv = CMath::dot(grad_w, grad_w, num_feat)+ grad_b*grad_b;
	}

	return dir_deriv;
}

DREAL CSubGradientSVM::compute_objective(INT num_feat, INT num_vec)
{
	DREAL result= 0.5 * CMath::dot(w,w, num_feat);
	
	for (INT i=0; i<num_vec; i++)
	{
		if (1 > proj[i])
			result += C1 * (1-proj[i]);
	}

	return result;
}

void CSubGradientSVM::update_projection(INT num_feat, INT num_vec)
{
	for (INT i=0; i<num_vec; i++)
		proj[i]=features->dense_dot(get_label(i), i, w, num_feat, bias);
}

void CSubGradientSVM::init(INT num_vec, INT num_feat)
{
	// alloc normal and bias inited with 0
	delete[] w;
	w=new DREAL[num_feat];
	ASSERT(w);
	memset(w,0,sizeof(DREAL)*num_feat);
	bias=0;

	proj= new DREAL[num_vec];
	ASSERT(proj);
	memset(proj,0,sizeof(DREAL)*num_vec);

	active=new BYTE[num_vec];
	ASSERT(active);
	memset(active,1,sizeof(BYTE)*num_vec);

	old_active=new BYTE[num_vec];
	ASSERT(old_active);
	memset(old_active,1,sizeof(BYTE)*num_vec);

	idx_bound=new INT[num_vec];
	ASSERT(idx_bound);
	memset(idx_bound,0,sizeof(INT)*num_vec);

	idx_active=new INT[num_vec];
	ASSERT(idx_active);
	memset(idx_active,0,sizeof(INT)*num_vec);

	sum_CXy_active=new DREAL[num_feat];
	ASSERT(sum_CXy_active);
	memset(sum_CXy_active,0,sizeof(DREAL)*num_feat);
}

void CSubGradientSVM::cleanup()
{
	delete[] proj;
	delete[] active;
	delete[] old_active;
	delete[] idx_bound;
	delete[] idx_active;
	delete[] sum_CXy_active;

	proj=NULL;
	active=NULL;
	old_active=NULL;
	idx_bound=NULL;
	idx_active=NULL;
	sum_CXy_active=NULL;
}

bool CSubGradientSVM::train()
{
	ASSERT(get_labels());
	ASSERT(get_features());

	INT num_train_labels=get_labels()->get_num_labels();
	INT num_feat=features->get_num_features();
	INT num_vec=features->get_num_vectors();

	ASSERT(num_vec==num_train_labels);

	init(num_vec, num_feat);

	INT num_active=0;
	INT num_bound=0;
	DREAL alpha=0.001; //learn rate

	update_projection(num_feat, num_vec);

	while (find_active(num_feat, num_vec, num_active, num_bound) > 0)
	{
		DREAL dir_deriv=compute_min_subgradient(num_feat, num_vec, num_active, num_bound);
		alpha=line_search();

		CMath::vec1_plus_scalar_times_vec2(-alpha, w, grad_w, num_feat);
		bias-=alpha*grad_b;

		update_projection(num_feat, num_vec);
		SG_PRINT("objective: %f dir_deriv: %f\n", compute_objective(num_feat, num_vec), dir_deriv);
	}

	cleanup();

	return true;
}
