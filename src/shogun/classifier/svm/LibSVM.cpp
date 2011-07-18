/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 1999-2009 Soeren Sonnenburg
 * Copyright (C) 1999-2009 Fraunhofer Institute FIRST and Max-Planck-Society
 */

#include "classifier/svm/LibSVM.h"
#include "lib/io.h"

using namespace shogun;

CLibSVM::CLibSVM(LIBSVM_SOLVER_TYPE st)
: CSVM(), model(NULL), solver_type(st)
{
}

CLibSVM::CLibSVM(float64_t C, CKernel* k, CLabels* lab)
: CSVM(C, k, lab), model(NULL), solver_type(LIBSVM_C_SVC)
{
	problem = svm_problem();
}

CLibSVM::~CLibSVM()
{
}


bool CLibSVM::train(CFeatures* data)
{
	struct svm_node* x_space;

	ASSERT(labels && labels->get_num_labels());
	ASSERT(labels->is_two_class_labeling());

	if (data)
	{
		if (labels->get_num_labels() != data->get_num_vectors())
			SG_ERROR("Number of training vectors does not match number of labels\n");
		kernel->init(data, data);
	}

	problem.l=labels->get_num_labels();
	SG_INFO( "%d trainlabels\n", problem.l);

	// set linear term
	if (m_linear_term_len > 0)
	{
		if (labels->get_num_labels() != m_linear_term_len)
            SG_ERROR("Number of training vectors does not match length of linear term\n");

		// set with linear term from base class
		problem.pv = get_linear_term_array();
	}
	else
	{
		// fill with minus ones
		problem.pv = new float64_t[problem.l];

		for (int i=0; i!=problem.l; i++)
			problem.pv[i] = -1.0;
	}

	problem.y=new float64_t[problem.l];
	problem.x=new struct svm_node*[problem.l];
    problem.C=new float64_t[problem.l];

	x_space=new struct svm_node[2*problem.l];

	for (int32_t i=0; i<problem.l; i++)
	{
		problem.y[i]=labels->get_label(i);
		problem.x[i]=&x_space[2*i];
		x_space[2*i].index=i;
		x_space[2*i+1].index=-1;
	}

	int32_t weights_label[2]={-1,+1};
	float64_t weights[2]={1.0,get_C2()/get_C1()};

	ASSERT(kernel && kernel->has_features());
    ASSERT(kernel->get_num_vec_lhs()==problem.l);

	param.svm_type=solver_type; // C SVM or NU_SVM
	param.kernel_type = LINEAR;
	param.degree = 3;
	param.gamma = 0;	// 1/k
	param.coef0 = 0;
	param.nu = get_nu();
	param.kernel=kernel;
	param.cache_size = kernel->get_cache_size();
	param.max_train_time = max_train_time;
	param.C = get_C1();
	param.eps = epsilon;
	param.p = 0.1;
	param.shrinking = 1;
	param.nr_weight = 2;
	param.weight_label = weights_label;
	param.weight = weights;
	param.use_bias = get_bias_enabled();

	const char* error_msg = svm_check_parameter(&problem, &param);

	if(error_msg)
		SG_ERROR("Error: %s\n",error_msg);

	model = svm_train(&problem, &param);

	if (model)
	{
		ASSERT(model->nr_class==2);
		ASSERT((model->l==0) || (model->l>0 && model->SV && model->sv_coef && model->sv_coef[0]));

		int32_t num_sv=model->l;

		create_new_model(num_sv);
		CSVM::set_objective(model->objective);

		float64_t sgn=model->label[0];

		set_bias(-sgn*model->rho[0]);

		for (int32_t i=0; i<num_sv; i++)
		{
			set_support_vector(i, (model->SV[i])->index);
			set_alpha(i, sgn*model->sv_coef[0][i]);
		}

		delete[] problem.x;
		delete[] problem.y;
		delete[] problem.pv;
        delete[] problem.C;


		delete[] x_space;

		svm_destroy_model(model);
		model=NULL;
		return true;
	}
	else
		return false;
}
