/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 1999-2009 Soeren Sonnenburg
 * Copyright (C) 1999-2009 Fraunhofer Institute FIRST and Max-Planck-Society
 */

#include <multiclass/MulticlassLibSVM.h>
#include <multiclass/MulticlassOneVsOneStrategy.h>
#include <labels/MulticlassLabels.h>
#include <io/SGIO.h>

using namespace shogun;

CMulticlassLibSVM::CMulticlassLibSVM(LIBSVM_SOLVER_TYPE st)
: CMulticlassSVM(new CMulticlassOneVsOneStrategy()), model(NULL), solver_type(st)
{
}

CMulticlassLibSVM::CMulticlassLibSVM(float64_t C, CKernel* k, CLabels* lab)
: CMulticlassSVM(new CMulticlassOneVsOneStrategy(), C, k, lab), model(NULL), solver_type(LIBSVM_C_SVC)
{
}

CMulticlassLibSVM::~CMulticlassLibSVM()
{
}

bool CMulticlassLibSVM::train_machine(CFeatures* data)
{
	struct svm_node* x_space;

	problem = svm_problem();

	ASSERT(m_labels && m_labels->get_num_labels())
	ASSERT(m_labels->get_label_type() == LT_MULTICLASS)
	int32_t num_classes = m_multiclass_strategy->get_num_classes();
	problem.l=m_labels->get_num_labels();
	SG_INFO("%d trainlabels, %d classes\n", problem.l, num_classes)


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
		problem.y[i]=((CMulticlassLabels*) m_labels)->get_label(i);
		problem.x[i]=&x_space[2*i];
		x_space[2*i].index=i;
		x_space[2*i+1].index=-1;
	}

	ASSERT(m_kernel)

	param.svm_type=solver_type; // C SVM or NU_SVM
	param.kernel_type = LINEAR;
	param.degree = 3;
	param.gamma = 0;	// 1/k
	param.coef0 = 0;
	param.nu = get_nu(); // Nu
	param.kernel=m_kernel;
	param.cache_size = m_kernel->get_cache_size();
	param.max_train_time = m_max_train_time;
	param.C = get_C();
	param.eps = get_epsilon();
	param.p = 0.1;
	param.shrinking = 1;
	param.nr_weight = 0;
	param.weight_label = NULL;
	param.weight = NULL;
	param.use_bias = svm_proto()->get_bias_enabled();

	const char* error_msg = svm_check_parameter(&problem,&param);

	if(error_msg)
		SG_ERROR("Error: %s\n",error_msg)

	model = svm_train(&problem, &param);

	if (model)
	{
		if (model->nr_class!=num_classes)
		{
			SG_ERROR("LibSVM model->nr_class=%d while num_classes=%d\n",
					model->nr_class, num_classes);
		}
		ASSERT((model->l==0) || (model->l>0 && model->SV && model->sv_coef))
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

				ASSERT(num_sv>0)
				ASSERT(model->sv_coef[i] && model->sv_coef[j-1])

				CSVM* svm=new CSVM(num_sv);

				svm->set_bias(sgn*bias);

				int32_t sv_idx=0;
				for (k=0; k<model->nSV[i]; k++)
				{
					SG_DEBUG("setting SV[%d] to %d\n", sv_idx,
							model->SV[offsets[i]+k]->index);
					svm->set_support_vector(sv_idx, model->SV[offsets[i]+k]->index);
					svm->set_alpha(sv_idx, sgn*model->sv_coef[j-1][offsets[i]+k]);
					sv_idx++;
				}

				for (k=0; k<model->nSV[j]; k++)
				{
					SG_DEBUG("setting SV[%d] to %d\n", sv_idx,
							model->SV[offsets[i]+k]->index);
					svm->set_support_vector(sv_idx, model->SV[offsets[j]+k]->index);
					svm->set_alpha(sv_idx, sgn*model->sv_coef[i][offsets[j]+k]);
					sv_idx++;
				}

				int32_t idx=0;

				if (num_classes > 3)
				{
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
				}
				else if (num_classes == 3)
				{
					idx = model->label[j]+model->label[i] - 1;
				}
				else if (num_classes == 2)
				{
					idx = i;
				}
//
//				if (sgn>0)
//					idx=((num_classes-1)*model->label[i]+model->label[j])/2;
//				else
//					idx=((num_classes-1)*model->label[j]+model->label[i])/2;
//
				SG_DEBUG("svm[%d] has %d sv (total: %d), b=%f "
						"label:(%d,%d) -> svm[%d]\n",
						s, num_sv, model->l, bias, model->label[i],
						model->label[j], idx);

				REQUIRE(set_svm(idx, svm),"SVM set failed")
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

		return true;
	}
	else
		return false;
}

