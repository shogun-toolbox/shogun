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
#include "features/SparseFeatures.h"
#include "lib/io.h"

CLibLinear::CLibLinear(LIBLINEAR_LOSS l) : CSparseLinearClassifier()
{
	loss=l;
	use_bias=false;
	C1=1;
	C2=1;
}

CLibLinear::~CLibLinear()
{
}

bool CLibLinear::train()
{
	ASSERT(get_labels());
	ASSERT(get_features());
	CSparseFeatures<DREAL>* sfeat=(CSparseFeatures<DREAL>*) features;

	INT num_train_labels=get_labels()->get_num_labels();
	INT num_feat=features->get_num_features();
	INT num_vec=features->get_num_vectors();

	ASSERT(num_vec==num_train_labels);
	delete[] w;
	if (use_bias)
		w=new DREAL[num_feat+1];
	else
		w=new DREAL[num_feat+0];

	w_dim=num_feat;
	ASSERT(w);

	problem prob;
	if (use_bias)
	{
		prob.n=w_dim+1;
		memset(w, 0, sizeof(DREAL)*(w_dim+1));
	}
	else
	{
		prob.n=w_dim;
		memset(w, 0, sizeof(DREAL)*(w_dim+0));
	}
	prob.l=num_vec;
	prob.x=new feature_node*[prob.l];
	prob.y=new int[prob.l];
	feature_node* x_space= new feature_node[sfeat->get_num_nonzero_entries() + 2*num_vec];

	ASSERT(x_space);
	ASSERT(prob.y);
	ASSERT(prob.x);

	INT j=0;
	for (int i=0; i<prob.l; i++)
	{
		prob.y[i]=get_labels()->get_int_label(i);
		prob.x[i]=&x_space[j];

		bool vfree;
		INT dim;
		TSparseEntry<DREAL>* sv=sfeat->get_sparse_feature_vector(i, dim, vfree);

		for (INT k=0; k<dim; k++)
		{
			x_space[j].index = sv[k].feat_index+1;
			x_space[j].value = sv[k].entry;
			j++;
		}

		sfeat->free_sparse_feature_vector(sv, i, vfree);

		if (use_bias)
		{
			x_space[j].index=num_feat+1;
			x_space[j].value=1.0;
			j++;
		}

		x_space[j].value=NAN;
		x_space[j].index=-1;
		j++;
	}

	SG_INFO( "%d training points %d dims\n", prob.l, prob.n);

	function *fun_obj=NULL;

	switch (loss)
	{
		case LR:
			fun_obj=new l2_lr_fun(&prob, get_C1(), get_C2());
			break;
		case L2:
			fun_obj=new l2loss_svm_fun(&prob, get_C1(), get_C2());
			break;
		default:
			SG_ERROR("unknown loss\n");
			break;
	}

	if (fun_obj)
	{
		CTron tron_obj(fun_obj, epsilon);
		tron_obj.tron(w);
		DREAL sgn=prob.y[0];

		for (INT i=0; i<w_dim; i++)
			w[i]*=sgn;

		if (use_bias)
			set_bias(sgn*w[w_dim]);
		else
			set_bias(0);

		delete fun_obj;
	}

	return true;
}
