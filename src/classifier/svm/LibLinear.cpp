/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 2 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 1999-2007 Soeren Sonnenburg
 * Copyright (C) 1999-2007 Fraunhofer Institute FIRST and Max-Planck-Society
 */

#include "classifier/svm/LibLinear.h"
#include "classifier/svm/SVM_linear.h"
#include "classifier/svm/Tron.h"
#include "lib/io.h"

CLibLinear::CLibLinear(LIBLINEAR_LOSS l) : CSparseLinearClassifier()
{
	loss=l;
}

CLibLinear::~CLibLinear()
{
}

bool CLibLinear::train()
{
	ASSERT(get_labels());
	ASSERT(get_features());

	INT num_train_labels=get_labels()->get_num_labels();
	INT num_feat=features->get_num_features();
	INT num_vec=features->get_num_vectors();

	ASSERT(num_vec==num_train_labels);
	delete[] w;
	w=new DREAL[num_feat];
	w_dim=num_feat;
	ASSERT(w);

	problem prob;
	prob.n=w_dim;
	prob.l=num_vec;
	prob.x=new feature_node*[prob.l];
	prob.y=new int[prob.l];
	ASSERT(prob.y);
	ASSERT(prob.x);

	for (int i=0; i<prob.l; i++)
	{
		prob.y[i]=get_labels()->get_int_label(i);
		prob.x[i]=NULL; //&x_space[2*i];
	}

	SG_INFO( "%d trainlabels\n", prob.l);

	function *fun_obj=NULL;

	switch (loss)
	{
		case L2_LR:
			fun_obj=new l2_lr_fun(&prob, get_C1(), get_C2());
			break;
		case L2LOSS_SVM:
			fun_obj=new l2loss_svm_fun(&prob, get_C1(), get_C2());
			break;
		default:
			SG_ERROR("unknown loss\n");
			break;
	}

	if(fun_obj)
	{
		CTron tron_obj(fun_obj, epsilon);
		tron_obj.tron(w);
		delete fun_obj;
	}

//	const char* error_msg = svm_check_parameter(&problem,&param);
//
//	if(error_msg)
//	{
//		fprintf(stderr,"Error: %s\n",error_msg);
//		exit(1);
//	}
//
//	model = svm_train(&problem, &param);
//
//	if (model)
//	{
//		ASSERT(model->nr_class==2);
//		ASSERT( (model->l==0) || (model->l > 0 && model->SV && model->sv_coef && model->sv_coef[0]) );
//
//		int num_sv=model->l;
//
//		create_new_model(num_sv);
//		CSVM::set_objective(model->objective);
//
//		DREAL sgn=model->label[0];
//
//		set_bias(-sgn*model->rho[0]);
//
//		for (int i=0; i<num_sv; i++)
//		{
//			set_support_vector(i, (model->SV[i])->index);
//			set_alpha(i, sgn*model->sv_coef[0][i]);
//		}
//
//		delete[] problem.x;
//		delete[] problem.y;
//		delete[] x_space;
//
//		return true;
//	}
//	else
		return false;
}
