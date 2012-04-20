/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 1999-2009 Soeren Sonnenburg
 * Copyright (C) 1999-2009 Fraunhofer Institute FIRST and Max-Planck-Society
 */

#include <shogun/classifier/svm/LibSVMMulticlass.h>
#include <shogun/io/SGIO.h>

using namespace shogun;

CLibSVMMulticlass::CLibSVMMulticlass(LIBSVM_SOLVER_TYPE st)
: CMulticlassSVM(ONE_VS_ONE_STRATEGY), model(NULL), solver_type(st)
{
}

CLibSVMMulticlass::CLibSVMMulticlass(float64_t C, CKernel* k, CLabels* lab)
: CMulticlassSVM(ONE_VS_ONE_STRATEGY, C, k, lab), model(NULL), solver_type(LIBSVM_C_SVC)
{
}

CLibSVMMulticlass::~CLibSVMMulticlass()
{
	//SG_PRINT("deleting LibSVM\n");
}

bool CLibSVMMulticlass::train_machine(CFeatures* data)
{
	struct svm_node* x_space;

	problem = svm_problem();

	ASSERT(m_labels && m_labels->get_num_labels());
	int32_t num_classes = m_labels->get_num_classes();
	problem.l=m_labels->get_num_labels();
	SG_INFO( "%d trainlabels, %d classes\n", problem.l, num_classes);

	/* ensure that there are only positive labels, otherwise, train_machine
	 * will produce memory errors since svm index gets wrong */
	for (index_t i=0; i<m_labels->get_num_labels(); ++i)
	{
		if (m_labels->get_label(i)<0)
		{
			SG_ERROR("Only labels >= 0 allowed for %s::train_machine!\n",
					get_name());
		}
	}

	if (data)
	{
		if (m_labels->get_num_labels() != data->get_num_vectors())
		{
			SG_ERROR("Number of training vectors does not match number of "
					"labels\n");
		}
		m_kernel->init(data, data);
	}

	problem.y=SG_MALLOC(float64_t, problem.l);
	problem.x=SG_MALLOC(struct svm_node*, problem.l);
	problem.pv=SG_MALLOC(float64_t, problem.l);
	problem.C=SG_MALLOC(float64_t, problem.l);

	x_space=SG_MALLOC(struct svm_node, 2*problem.l);

	for (int32_t i=0; i<problem.l; i++)
	{
		problem.pv[i]=-1.0;
		problem.y[i]=m_labels->get_label(i);
		problem.x[i]=&x_space[2*i];
		x_space[2*i].index=i;
		x_space[2*i+1].index=-1;
	}

	ASSERT(m_kernel);

	param.svm_type=solver_type; // C SVM or NU_SVM
	param.kernel_type = LINEAR;
	param.degree = 3;
	param.gamma = 0;	// 1/k
	param.coef0 = 0;
	param.nu = get_nu(); // Nu
	param.kernel=m_kernel;
	param.cache_size = m_kernel->get_cache_size();
	param.max_train_time = m_max_train_time;
	param.C = get_C1();
	param.eps = get_epsilon();
	param.p = 0.1;
	param.shrinking = 1;
	param.nr_weight = 0;
	param.weight_label = NULL;
	param.weight = NULL;
	param.use_bias = svm_proto()->get_bias_enabled();

	const char* error_msg = svm_check_parameter(&problem,&param);

	if(error_msg)
		SG_ERROR("Error: %s\n",error_msg);

	model = svm_train(&problem, &param);

	if (model)
	{
		if (model->nr_class!=num_classes)
		{
			SG_ERROR("LibSVM model->nr_class=%d while num_classes=%d\n",
					model->nr_class, num_classes);
		}
		ASSERT((model->l==0) || (model->l>0 && model->SV && model->sv_coef));
		create_multiclass_svm(num_classes);

		int32_t* offsets=SG_MALLOC(int32_t, num_classes);
		offsets[0]=0;

		for (int32_t i=1; i<num_classes; i++)
			offsets[i] = offsets[i-1]+model->nSV[i-1];

		int32_t s=0;
		for (int32_t i=0; i<num_classes; i++)
		{
			for (int32_t j=i+1; j<num_classes; j++)
			{
				int32_t k, l;

				float64_t sgn=1;
				if (model->label[i]>model->label[j])
					sgn=-1;

				int32_t num_sv=model->nSV[i]+model->nSV[j];
				float64_t bias=-model->rho[s];

				ASSERT(num_sv>0);
				ASSERT(model->sv_coef[i] && model->sv_coef[j-1]);

				CSVM* svm=new CSVM(num_sv);

				svm->set_bias(sgn*bias);

				int32_t sv_idx=0;
				for (k=0; k<model->nSV[i]; k++)
				{
					svm->set_support_vector(sv_idx, model->SV[offsets[i]+k]->index);
					svm->set_alpha(sv_idx, sgn*model->sv_coef[j-1][offsets[i]+k]);
					sv_idx++;
				}

				for (k=0; k<model->nSV[j]; k++)
				{
					svm->set_support_vector(sv_idx, model->SV[offsets[j]+k]->index);
					svm->set_alpha(sv_idx, sgn*model->sv_coef[i][offsets[j]+k]);
					sv_idx++;
				}

				int32_t idx=0;

				if (sgn>0)
				{
					for (k=0; k<model->label[i]; k++)
						idx+=num_classes-k-1;

					for (l=model->label[i]+1; l<model->label[j]; l++)
						idx++;
				}
				else
				{
					for (k=0; k<model->label[j]; k++)
						idx+=num_classes-k-1;

					for (l=model->label[j]+1; l<model->label[i]; l++)
						idx++;
				}


//				if (sgn>0)
//					idx=((num_classes-1)*model->label[i]+model->label[j])/2;
//				else
//					idx=((num_classes-1)*model->label[j]+model->label[i])/2;
//
				SG_DEBUG("svm[%d] has %d sv (total: %d), b=%f label:(%d,%d) -> svm[%d]\n", s, num_sv, model->l, bias, model->label[i], model->label[j], idx);
				set_svm(idx, svm);
				s++;
			}
		}

		set_objective(model->objective);

		SG_FREE(offsets);
		SG_FREE(problem.x);
		SG_FREE(problem.y);
		SG_FREE(x_space);
		SG_FREE(problem.pv);
		SG_FREE(problem.C);

		svm_destroy_model(model);
		model=NULL;

		/* the features needed for the model are all support vectors for now,
		 * which  means that a copy of the features is stored in lhs */
		/* TODO this can be done better, ie only store sv of underlying svms
		 * and map indices */
		svm_svs().destroy_vector();
		svm_svs()=SGVector<index_t>(m_kernel->get_num_vec_lhs());
		svm_svs().range_fill();

		return true;
	}
	else
		return false;
}

