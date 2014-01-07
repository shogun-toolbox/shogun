/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2006 Christian Gehl
 * Written (W) 2006-2009 Soeren Sonnenburg
 * Copyright (C) 1999-2009 Fraunhofer Institute FIRST and Max-Planck-Society
 */

#include <classifier/svm/LibSVMOneClass.h>
#include <io/SGIO.h>

using namespace shogun;

CLibSVMOneClass::CLibSVMOneClass()
: CSVM(), model(NULL)
{
}

CLibSVMOneClass::CLibSVMOneClass(float64_t C, CKernel* k)
: CSVM(C, k, NULL), model(NULL)
{
}

CLibSVMOneClass::~CLibSVMOneClass()
{
	SG_FREE(model);
}

bool CLibSVMOneClass::train_machine(CFeatures* data)
{
	ASSERT(kernel)
	if (data)
		kernel->init(data, data);

	problem.l=kernel->get_num_vec_lhs();

	struct svm_node* x_space;
	SG_INFO("%d train data points\n", problem.l)

	problem.y=NULL;
	problem.x=SG_MALLOC(struct svm_node*, problem.l);
	x_space=SG_MALLOC(struct svm_node, 2*problem.l);

	for (int32_t i=0; i<problem.l; i++)
	{
		problem.x[i]=&x_space[2*i];
		x_space[2*i].index=i;
		x_space[2*i+1].index=-1;
	}

	int32_t weights_label[2]={-1,+1};
	float64_t weights[2]={1.0,get_C2()/get_C1()};

	param.svm_type=ONE_CLASS; // C SVM
	param.kernel_type = LINEAR;
	param.degree = 3;
	param.gamma = 0;	// 1/k
	param.coef0 = 0;
	param.nu = get_nu();
	param.kernel=kernel;
	param.cache_size = kernel->get_cache_size();
	param.max_train_time = m_max_train_time;
	param.C = get_C1();
	param.eps = epsilon;
	param.p = 0.1;
	param.shrinking = 1;
	param.nr_weight = 2;
	param.weight_label = weights_label;
	param.weight = weights;
	param.use_bias = get_bias_enabled();
	
	const char* error_msg = svm_check_parameter(&problem,&param);

	if(error_msg)
		SG_ERROR("Error: %s\n",error_msg)
	
	model = svm_train(&problem, &param);

	if (model)
	{
		ASSERT(model->nr_class==2)
		ASSERT((model->l==0) || (model->l>0 && model->SV && model->sv_coef && model->sv_coef[0]))

		int32_t num_sv=model->l;

		create_new_model(num_sv);
		CSVM::set_objective(model->objective);

		set_bias(-model->rho[0]);
		for (int32_t i=0; i<num_sv; i++)
		{
			set_support_vector(i, (model->SV[i])->index);
			set_alpha(i, model->sv_coef[0][i]);
		}

		SG_FREE(problem.x);
		SG_FREE(x_space);
		svm_destroy_model(model);
		model=NULL;

		return true;
	}
	else
		return false;
}
