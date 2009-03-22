/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 1999-2009 Soeren Sonnenburg
 * Copyright (C) 1999-2009 Fraunhofer Institute FIRST and Max-Planck-Society
 */
#include "lib/config.h"

#ifdef HAVE_LAPACK
#include "lib/io.h"
#include "classifier/svm/LibLinear.h"
#include "classifier/svm/SVM_linear.h"
#include "classifier/svm/Tron.h"
#include "features/DotFeatures.h"

CLibLinear::CLibLinear(LIBLINEAR_LOSS l)
: CLinearClassifier()
{
	loss=l;
	use_bias=false;
	C1=1;
	C2=1;
}

CLibLinear::CLibLinear(
	float64_t C, CDotFeatures* traindat, CLabels* trainlab)
: CLinearClassifier(), C1(C), C2(C), use_bias(true), epsilon(1e-5)
{
	set_features(traindat);
	set_labels(trainlab);
	loss=LR;
}


CLibLinear::~CLibLinear()
{
}

bool CLibLinear::train()
{
	ASSERT(labels);
	ASSERT(features);
	ASSERT(labels->is_two_class_labeling());

	int32_t num_train_labels=labels->get_num_labels();
	int32_t num_feat=features->get_dim_feature_space();
	int32_t num_vec=features->get_num_vectors();

	ASSERT(num_vec==num_train_labels);
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
		float64_t sgn=prob.y[0];

		for (int32_t i=0; i<w_dim; i++)
			w[i]*=sgn;

		if (use_bias)
			set_bias(sgn*w[w_dim]);
		else
			set_bias(0);

		delete fun_obj;
	}

    delete[] prob.y;

	return true;
}
#endif //HAVE_LAPACK
