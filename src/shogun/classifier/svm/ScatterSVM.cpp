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
#ifdef USE_SVMLIGHT
#include <shogun/classifier/svm/SVMLightOneClass.h>
#endif //USE_SVMLIGHT

#include <shogun/kernel/Kernel.h>
#include <shogun/classifier/svm/ScatterSVM.h>
#include <shogun/kernel/ScatterKernelNormalizer.h>
#include <shogun/io/SGIO.h>

using namespace shogun;

CScatterSVM::CScatterSVM(void)
: CMultiClassSVM(ONE_VS_REST), scatter_type(NO_BIAS_LIBSVM),
  model(NULL), norm_wc(NULL), norm_wcw(NULL), rho(0), m_num_classes(0)
{
	SG_UNSTABLE("CScatterSVM::CScatterSVM(void)", "\n");
}

CScatterSVM::CScatterSVM(SCATTER_TYPE type)
: CMultiClassSVM(ONE_VS_REST), scatter_type(type), model(NULL),
	norm_wc(NULL), norm_wcw(NULL), rho(0), m_num_classes(0)
{
}

CScatterSVM::CScatterSVM(float64_t C, CKernel* k, CLabels* lab)
: CMultiClassSVM(ONE_VS_REST, C, k, lab), scatter_type(NO_BIAS_LIBSVM), model(NULL),
	norm_wc(NULL), norm_wcw(NULL), rho(0), m_num_classes(0)
{
}

CScatterSVM::~CScatterSVM()
{
	SG_FREE(norm_wc);
	SG_FREE(norm_wcw);
}

bool CScatterSVM::train_kernel_machine(CFeatures* data)
{
	ASSERT(labels && labels->get_num_labels());
	m_num_classes = labels->get_num_classes();
	int32_t num_vectors = labels->get_num_labels();

	if (data)
	{
		if (labels->get_num_labels() != data->get_num_vectors())
			SG_ERROR("Number of training vectors does not match number of labels\n");
		kernel->init(data, data);
	}

	int32_t* numc=new int32_t[m_num_classes];
	CMath::fill_vector(numc, m_num_classes, 0);

	for (int32_t i=0; i<num_vectors; i++)
		numc[(int32_t) labels->get_int_label(i)]++;

	int32_t Nc=0;
	int32_t Nmin=num_vectors;
	for (int32_t i=0; i<m_num_classes; i++)
	{
		if (numc[i]>0)
		{
			Nc++;
			Nmin=CMath::min(Nmin, numc[i]);
		}

	}
	SG_FREE(numc);
	m_num_classes=m_num_classes;

	bool result=false;

	if (scatter_type==NO_BIAS_LIBSVM)
	{
		result=train_no_bias_libsvm();
	}
#ifdef USE_SVMLIGHT
	else if (scatter_type==NO_BIAS_SVMLIGHT)
	{
		result=train_no_bias_svmlight();
	}
#endif //USE_SVMLIGHT
	else if (scatter_type==TEST_RULE1 || scatter_type==TEST_RULE2) 
	{
		float64_t nu_min=((float64_t) Nc)/num_vectors;
		float64_t nu_max=((float64_t) Nc)*Nmin/num_vectors;

		SG_INFO("valid nu interval [%f ... %f]\n", nu_min, nu_max);

		if (get_nu()<nu_min || get_nu()>nu_max)
			SG_ERROR("nu out of valid range [%f ... %f]\n", nu_min, nu_max);

		result=train_testrule12();
	}
	else
		SG_ERROR("Unknown Scatter type\n"); 

	return result;
}

bool CScatterSVM::train_no_bias_libsvm()
{
	struct svm_node* x_space;

	problem.l=labels->get_num_labels();
	SG_INFO( "%d trainlabels\n", problem.l);

	problem.y=new float64_t[problem.l];
	problem.x=new struct svm_node*[problem.l];
	x_space=new struct svm_node[2*problem.l];

	for (int32_t i=0; i<problem.l; i++)
	{
		problem.y[i]=+1;
		problem.x[i]=&x_space[2*i];
		x_space[2*i].index=i;
		x_space[2*i+1].index=-1;
	}

	int32_t weights_label[2]={-1,+1};
	float64_t weights[2]={1.0,get_C2()/get_C1()};

	ASSERT(kernel && kernel->has_features());
    ASSERT(kernel->get_num_vec_lhs()==problem.l);

	param.svm_type=C_SVC; // Nu MC SVM
	param.kernel_type = LINEAR;
	param.degree = 3;
	param.gamma = 0;	// 1/k
	param.coef0 = 0;
	param.nu = get_nu(); // Nu
	CKernelNormalizer* prev_normalizer=kernel->get_normalizer();
	kernel->set_normalizer(new CScatterKernelNormalizer(
				m_num_classes-1, -1, labels, prev_normalizer));
	param.kernel=kernel;
	param.cache_size = kernel->get_cache_size();
	param.C = 0;
	param.eps = epsilon;
	param.p = 0.1;
	param.shrinking = 0;
	param.nr_weight = 2;
	param.weight_label = weights_label;
	param.weight = weights;
	param.nr_class=m_num_classes;
	param.use_bias = get_bias_enabled();

	const char* error_msg = svm_check_parameter(&problem,&param);

	if(error_msg)
		SG_ERROR("Error: %s\n",error_msg);

	model = svm_train(&problem, &param);
	kernel->set_normalizer(prev_normalizer);
	SG_UNREF(prev_normalizer);

	if (model)
	{
		ASSERT((model->l==0) || (model->l>0 && model->SV && model->sv_coef && model->sv_coef));

		ASSERT(model->nr_class==m_num_classes);
		create_multiclass_svm(m_num_classes);

		rho=model->rho[0];

		SG_FREE(norm_wcw);
		norm_wcw = new float64_t[m_num_svms];

		for (int32_t i=0; i<m_num_classes; i++)
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

		SG_FREE(problem.x);
		SG_FREE(problem.y);
		SG_FREE(x_space);
		for (int32_t i=0; i<m_num_classes; i++)
		{
			SG_FREE(model->SV[i]);
			model->SV[i]=NULL;
		}
		svm_destroy_model(model);

		if (scatter_type==TEST_RULE2)
			compute_norm_wc();

		model=NULL;
		return true;
	}
	else
		return false;
}

#ifdef USE_SVMLIGHT
bool CScatterSVM::train_no_bias_svmlight()
{
	CKernelNormalizer* prev_normalizer=kernel->get_normalizer();
	CScatterKernelNormalizer* n=new CScatterKernelNormalizer(
				 m_num_classes-1, -1, labels, prev_normalizer);
	kernel->set_normalizer(n);
	kernel->init_normalizer();

	CSVMLightOneClass* light=new CSVMLightOneClass(C1, kernel);
	light->set_linadd_enabled(false);
	light->train();

	SG_FREE(norm_wcw);
	norm_wcw = new float64_t[m_num_classes];

	int32_t num_sv=light->get_num_support_vectors();
	create_new_model(num_sv);

	for (int32_t i=0; i<num_sv; i++)
	{
		set_alpha(i, light->get_alpha(i));
		set_support_vector(i, light->get_support_vector(i));
	}

	kernel->set_normalizer(prev_normalizer);
	return true;
}
#endif //USE_SVMLIGHT

bool CScatterSVM::train_testrule12()
{
	struct svm_node* x_space;
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
	param.nr_class=m_num_classes;
	param.use_bias = get_bias_enabled();

	const char* error_msg = svm_check_parameter(&problem,&param);

	if(error_msg)
		SG_ERROR("Error: %s\n",error_msg);

	model = svm_train(&problem, &param);

	if (model)
	{
		ASSERT((model->l==0) || (model->l>0 && model->SV && model->sv_coef && model->sv_coef));

		ASSERT(model->nr_class==m_num_classes);
		create_multiclass_svm(m_num_classes);

		rho=model->rho[0];

		SG_FREE(norm_wcw);
		norm_wcw = new float64_t[m_num_svms];

		for (int32_t i=0; i<m_num_classes; i++)
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

		SG_FREE(problem.x);
		SG_FREE(problem.y);
		SG_FREE(x_space);
		for (int32_t i=0; i<m_num_classes; i++)
		{
			SG_FREE(model->SV[i]);
			model->SV[i]=NULL;
		}
		svm_destroy_model(model);

		if (scatter_type==TEST_RULE2)
			compute_norm_wc();

		model=NULL;
		return true;
	}
	else
		return false;
}

void CScatterSVM::compute_norm_wc()
{
	SG_FREE(norm_wc);
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

CLabels* CScatterSVM::classify_one_vs_rest()
{
	CLabels* output=NULL;
	if (!kernel)
	{
		SG_ERROR( "SVM can not proceed without kernel!\n");
		return false ;
	}

	if ( kernel && kernel->get_num_vec_lhs() && kernel->get_num_vec_rhs())
	{
		int32_t num_vectors=kernel->get_num_vec_rhs();

		output=new CLabels(num_vectors);
		SG_REF(output);

		if (scatter_type == TEST_RULE1)
		{
			ASSERT(m_num_svms>0);
			for (int32_t i=0; i<num_vectors; i++)
				output->set_label(i, apply(i));
		}
#ifdef USE_SVMLIGHT
		else if (scatter_type == NO_BIAS_SVMLIGHT)
		{
			float64_t* outputs=new float64_t[num_vectors*m_num_classes];
			CMath::fill_vector(outputs,num_vectors*m_num_classes,0.0);

			for (int32_t i=0; i<num_vectors; i++)
			{
				for (int32_t j=0; j<get_num_support_vectors(); j++)
				{
					float64_t score=kernel->kernel(get_support_vector(j), i)*get_alpha(j);
					int32_t label=labels->get_int_label(get_support_vector(j));
					for (int32_t c=0; c<m_num_classes; c++)
					{
						float64_t s= (label==c) ? (m_num_classes-1) : (-1);
						outputs[c+i*m_num_classes]+=s*score;
					}
				}
			}

			for (int32_t i=0; i<num_vectors; i++)
			{
				int32_t winner=0;
				float64_t max_out=outputs[i*m_num_classes+0];

				for (int32_t j=1; j<m_num_classes; j++)
				{
					float64_t out=outputs[i*m_num_classes+j];

					if (out>max_out)
					{
						winner=j;
						max_out=out;
					}
				}

				output->set_label(i, winner);
			}

			SG_FREE(outputs);
		}
#endif //USE_SVMLIGHT
		else
		{
			ASSERT(m_num_svms>0);
			ASSERT(num_vectors==output->get_num_labels());
			CLabels** outputs=new CLabels*[m_num_svms];

			for (int32_t i=0; i<m_num_svms; i++)
			{
				//SG_PRINT("svm %d\n", i);
				ASSERT(m_svms[i]);
				m_svms[i]->set_kernel(kernel);
				m_svms[i]->set_labels(labels);
				outputs[i]=m_svms[i]->apply();
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

			SG_FREE(outputs);
		}
	}

	return output;
}

float64_t CScatterSVM::apply(int32_t num)
{
	ASSERT(m_num_svms>0);
	float64_t* outputs=new float64_t[m_num_svms];
	int32_t winner=0;

	if (scatter_type == TEST_RULE1)
	{
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
	}
#ifdef USE_SVMLIGHT
	else if (scatter_type == NO_BIAS_SVMLIGHT)
	{
		SG_ERROR("Use classify...\n");
	}
#endif //USE_SVMLIGHT
	else
	{
		float64_t max_out=m_svms[0]->apply(num)/norm_wc[0];

		for (int32_t i=1; i<m_num_svms; i++)
		{
			outputs[i]=m_svms[i]->apply(num)/norm_wc[i];
			if (outputs[i]>max_out)
			{
				winner=i;
				max_out=outputs[i];
			}
		}
	}

	SG_FREE(outputs);
	return winner;
}
