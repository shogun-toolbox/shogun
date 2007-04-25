/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 2 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 1999-2007 Soeren Sonnenburg
 * Copyright (C) 1999-2007 Fraunhofer Institute FIRST and Max-Planck-Society
 */

#include "classifier/svm/LibSVM_multiclass.h"
#include "lib/io.h"

CLibSVMMultiClass::CLibSVMMultiClass() : CMultiClassSVM(ONE_VS_ONE), model(NULL)
{
}

CLibSVMMultiClass::CLibSVMMultiClass(DREAL C, CKernel* k, CLabels* lab) : CMultiClassSVM(ONE_VS_ONE, C, k, lab), model(NULL)
{
}

CLibSVMMultiClass::~CLibSVMMultiClass()
{
	//SG_PRINT("deleting LibSVM\n");
}

bool CLibSVMMultiClass::train()
{
	free(model);

	struct svm_node* x_space;

	ASSERT(get_labels() && get_labels()->get_num_labels());
	INT num_classes = get_labels()->get_num_classes();
	problem.l=get_labels()->get_num_labels();
	SG_INFO( "%d trainlabels, %d classes\n", problem.l, num_classes);

	problem.y=new double[problem.l];
	problem.x=new struct svm_node*[problem.l];
	x_space=new struct svm_node[2*problem.l];

	ASSERT(problem.y);
	ASSERT(problem.x);
	ASSERT(x_space);

	for (int i=0; i<problem.l; i++)
	{
		problem.y[i]=get_labels()->get_label(i);
		problem.x[i]=&x_space[2*i];
		x_space[2*i].index=i;
		x_space[2*i+1].index=-1;
	}

	ASSERT(get_kernel());

	param.svm_type=C_SVC; // C SVM
	param.kernel_type = LINEAR;
	param.degree = 3;
	param.gamma = 0;	// 1/k
	param.coef0 = 0;
	param.nu = 0.5;
	param.kernel=get_kernel();
	param.cache_size = get_kernel()->get_cache_size();
	param.C = get_C1();
	param.eps = epsilon;
	param.p = 0.1;
	param.shrinking = 1;
	param.nr_weight = 0;
	param.weight_label = NULL;
	param.weight = NULL;

	const char* error_msg = svm_check_parameter(&problem,&param);

	if(error_msg)
	{
		fprintf(stderr,"Error: %s\n",error_msg);
		exit(1);
	}

	model = svm_train(&problem, &param);

	if (model)
	{
		ASSERT(model->nr_class==num_classes);
		ASSERT( (model->l==0) || (model->l > 0 && model->SV && model->sv_coef) );
		create_multiclass_svm(num_classes);

		INT* offsets=new INT[num_classes];
		ASSERT(offsets);
		offsets[0]=0;
		
		for (INT i=1; i<num_classes; i++)
			offsets[i] = offsets[i-1]+model->nSV[i-1];

		INT s=0;
		for (INT i=0; i<num_classes; i++)
		{
			for (INT j=i+1; j<num_classes; j++)
			{
				DREAL sgn=1;
				if (model->label[i]>model->label[j])
					sgn=-1;

				int num_sv=model->nSV[i]+model->nSV[j];
				DREAL bias=-model->rho[s];

				ASSERT(num_sv>0);
				ASSERT(model->sv_coef[i] && model->sv_coef[j-1]);

				CSVM* svm=new CSVM(num_sv);

				svm->set_bias(sgn*bias);

				INT sv_idx=0;
				for (int k=0; k<model->nSV[i]; k++)
				{
					svm->set_support_vector(sv_idx, model->SV[offsets[i]+k]->index);
					svm->set_alpha(sv_idx, sgn*model->sv_coef[j-1][offsets[i]+k]);
					sv_idx++;
				}

				for (int k=0; k<model->nSV[j]; k++)
				{
					svm->set_support_vector(sv_idx, model->SV[offsets[j]+k]->index);
					svm->set_alpha(sv_idx, sgn*model->sv_coef[i][offsets[j]+k]);
					sv_idx++;
				}

				INT idx=0;

				if (sgn>0)
				{
					for (INT k=0; k<model->label[i]; k++)
						idx+=num_classes-k-1;

					for (INT l=model->label[i]+1; l<model->label[j]; l++)
						idx++;
				}
				else
				{
					for (INT k=0; k<model->label[j]; k++)
						idx+=num_classes-k-1;

					for (INT l=model->label[j]+1; l<model->label[i]; l++)
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

		CSVM::set_objective(model->objective);

		delete[] problem.x;
		delete[] problem.y;
		delete[] x_space;

		return true;
	}
	else
		return false;
}

