/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2009 Soeren Sonnenburg
 * Written (W) 2009 Marius Kloft
 * Copyright (C) 2009 TU Berlin and Max-Planck-Society
 */

#include "classifier/svm/MCSVM.h"
#include "lib/io.h"

CMCSVM::CMCSVM()
: CMultiClassSVM(ONE_VS_REST), model(NULL), norm_wc(NULL), norm_wcw(NULL), rho(0)
{
}

CMCSVM::CMCSVM(float64_t C, CKernel* k, CLabels* lab)
: CMultiClassSVM(ONE_VS_REST, C, k, lab), model(NULL), norm_wc(NULL), norm_wcw(NULL), rho(0)
{
}

CMCSVM::~CMCSVM()
{
	delete[] norm_wc;
	delete[] norm_wcw;
	//SG_PRINT("deleting MCSVM\n");
}

bool CMCSVM::train(CFeatures* data)
{
	struct svm_node* x_space;

	ASSERT(labels && labels->get_num_labels());
	int32_t num_classes = labels->get_num_classes();

	if (data)
	{
		if (labels->get_num_labels() != data->get_num_vectors())
			SG_ERROR("Number of training vectors does not match number of labels\n");
		kernel->init(data, data);
	}

	problem.l=labels->get_num_labels();
	SG_INFO( "%d trainlabels\n", problem.l);

	problem.y=new float64_t[problem.l];
	problem.x=new struct svm_node*[problem.l];
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

	param.svm_type=NU_MULTICLASS_SVC; // Nu MC SVM
	param.kernel_type = LINEAR;
	param.degree = 3;
	param.gamma = 0;	// 1/k
	param.coef0 = 0;
	param.nu = get_nu(); // Nu
	param.kernel=kernel;
	param.cache_size = kernel->get_cache_size();
	param.C = 0;
	param.eps = epsilon;
	param.p = 0.1;
	param.shrinking = 0;
	param.nr_weight = 2;
	param.weight_label = weights_label;
	param.weight = weights;
	param.nr_class=num_classes;
	param.use_bias = get_bias_enabled();

	int32_t* numc=new int32_t[num_classes];
	CMath::fill_vector(numc, num_classes, 0);

	for (int32_t i=0; i<problem.l; i++)
		numc[(int32_t) problem.y[i]]++;

	int32_t Nc=0;
	int32_t Nmin=problem.l;
	for (int32_t i=0; i<num_classes; i++)
	{
		if (numc[i]>0)
		{
			Nc++;
			Nmin=CMath::min(Nmin, numc[i]);
		}

	}

	float64_t nu_min=((float64_t) Nc)/problem.l;
	float64_t nu_max=((float64_t) Nc)*Nmin/problem.l;

	SG_INFO("valid nu interval [%f ... %f]\n", nu_min, nu_max);

	if (param.nu<nu_min || param.nu>nu_max)
		SG_ERROR("nu out of valid range [%f ... %f]\n", nu_min, nu_max);

	const char* error_msg = svm_check_parameter(&problem,&param);

	if(error_msg)
		SG_ERROR("Error: %s\n",error_msg);

	model = svm_train(&problem, &param);

	if (model)
	{
		ASSERT((model->l==0) || (model->l>0 && model->SV && model->sv_coef && model->sv_coef));

		ASSERT(model->nr_class==num_classes);
		create_multiclass_svm(num_classes);

		rho=model->rho[0];

		delete[] norm_wcw;
		norm_wcw = new float64_t[m_num_svms];

		for (int32_t i=0; i<num_classes; i++)
		{
			int32_t num_sv=model->nSV[i];

			CSVM* svm=new CSVM(num_sv);
			svm->set_bias(model->rho[i+1]);
			norm_wcw[i]=model->normwcw[i];


			for (int32_t j=0; j<num_sv; j++)
			{
				svm->set_alpha(j, model->sv_coef[i][j]);
				svm->set_support_vector(j, model->SV[i][j].index);
			}

			set_svm(i, svm);
		}

		delete[] problem.x;
		delete[] problem.y;
		delete[] x_space;
		for (int32_t i=0; i<num_classes; i++)
		{
			free(model->SV[i]);
			model->SV[i]=NULL;
		}
		svm_destroy_model(model);
		compute_norm_wc();

		model=NULL;
		return true;
	}
	else
		return false;
}

void CMCSVM::compute_norm_wc()
{
	delete[] norm_wc;
	norm_wc = new float64_t[m_num_svms];
	for (int32_t i=0; i<m_num_svms; i++)
		norm_wc[i]=0;


	for (int c=0; c<m_num_svms; c++)
	{
		CSVM* svm=m_svms[c];
		int32_t num_sv = svm->get_num_support_vectors();

		for (int32_t i=0; i<num_sv; i++)
		{
			int32_t ii=svm->get_support_vector(i);
			for (int32_t j=0; j<num_sv; j++)
			{
				int32_t jj=svm->get_support_vector(j);
				norm_wc[c]+=svm->get_alpha(i)*kernel->kernel(ii,jj)*svm->get_alpha(j);
			}
		}
	}

	for (int32_t i=0; i<m_num_svms; i++)
		norm_wc[i]=CMath::sqrt(norm_wc[i]);

	CMath::display_vector(norm_wc, m_num_svms, "norm_wc");
}

CLabels* CMCSVM::classify_one_vs_rest(CLabels* output)
{
	ASSERT(m_num_svms>0);

	if (!kernel)
	{
		SG_ERROR( "SVM can not proceed without kernel!\n");
		return false ;
	}

	if ( kernel && kernel->get_num_vec_lhs() && kernel->get_num_vec_rhs())
	{
		int32_t num_vectors=kernel->get_num_vec_rhs();

		if (!output)
		{
			output=new CLabels(num_vectors);
			SG_REF(output);
		}

		for (int32_t i=0; i<num_vectors; i++)
		{
			output->set_label(i, classify_example(i));
		}
/*
		ASSERT(num_vectors==output->get_num_labels());
		CLabels** outputs=new CLabels*[m_num_svms];

		for (int32_t i=0; i<m_num_svms; i++)
		{
			ASSERT(m_svms[i]);
			m_svms[i]->set_kernel(kernel);
			m_svms[i]->set_labels(labels);
			outputs[i]=m_svms[i]->classify();
		}

		for (int32_t i=0; i<num_vectors; i++)
		{
			int32_t winner=0;
			float64_t max_out=outputs[0]->get_label(i)/norm_wc[0];

			for (int32_t j=1; j<m_num_svms; j++)
			{
				float64_t out=outputs[j]->get_label(i)/norm_wc[j];

				if (out>max_out)
				{
					winner=j;
					max_out=out;
				}
			}

			output->set_label(i, winner);
		}

		for (int32_t i=0; i<m_num_svms; i++)
			SG_UNREF(outputs[i]);

		delete[] outputs;
		*/
	}

	return output;
}

float64_t CMCSVM::classify_example(int32_t num)
{
	/*
	ASSERT(m_num_svms>0);
	float64_t* outputs=new float64_t[m_num_svms];
	int32_t winner=0;
	float64_t max_out=m_svms[0]->classify_example(num)/norm_wc[0];

	for (int32_t i=1; i<m_num_svms; i++)
	{
		outputs[i]=m_svms[i]->classify_example(num)/norm_wc[i];
		if (outputs[i]>max_out)
		{
			winner=i;
			max_out=outputs[i];
		}
	}
	delete[] outputs;

	return winner;
	*/

	ASSERT(m_num_svms>0);
	float64_t* outputs=new float64_t[m_num_svms];
	int32_t winner=0;

	for (int32_t c=0; c<m_num_svms; c++)
		outputs[c]=m_svms[c]->get_bias()-rho;

	for (int32_t c=0; c<m_num_svms; c++)
	{
		float64_t v=0;

		for (int32_t i=0; i<m_svms[c]->get_num_support_vectors(); i++)
		{
			float64_t alpha=m_svms[c]->get_alpha(i);
			int32_t svidx=m_svms[c]->get_support_vector(i);
			v += alpha*kernel->kernel(svidx, num);
		}

		outputs[c] += v;
		for (int32_t j=0; j<m_num_svms; j++)
			outputs[j] -= v/m_num_svms;
	}

	for (int32_t j=0; j<m_num_svms; j++)
		outputs[j]/=norm_wcw[j];

	float64_t max_out=outputs[0];
	for (int32_t j=0; j<m_num_svms; j++)
	{
		if (outputs[j]>max_out)
		{
			max_out=outputs[j];
			winner=j;
		}
	}

	delete[] outputs;

	//SG_PRINT("winner = %d\n", winner);

	return winner;
}
