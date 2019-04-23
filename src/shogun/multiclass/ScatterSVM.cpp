/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Soeren Sonnenburg, Sergey Lisitsyn, Chiyuan Zhang, Viktor Gal,
 *          Leon Kuchenbecker, Kyle McQuisten
 */
#include <shogun/multiclass/ScatterSVM.h>

#ifdef USE_SVMLIGHT
#include <shogun/classifier/svm/SVMLightOneClass.h>
#endif //USE_SVMLIGHT

#include <shogun/kernel/Kernel.h>
#include <shogun/kernel/normalizer/ScatterKernelNormalizer.h>
#include <shogun/multiclass/MulticlassOneVsRestStrategy.h>
#include <shogun/io/SGIO.h>

using namespace shogun;

ScatterSVM::ScatterSVM()
: MulticlassSVM(std::make_shared<MulticlassOneVsRestStrategy>()), scatter_type(NO_BIAS_LIBSVM),
  norm_wc(NULL), norm_wc_len(0), norm_wcw(NULL), norm_wcw_len(0), rho(0), m_num_classes(0)
{
	SG_UNSTABLE("ScatterSVM::ScatterSVM()", "\n")
}

ScatterSVM::ScatterSVM(SCATTER_TYPE type)
: MulticlassSVM(std::make_shared<MulticlassOneVsRestStrategy>()), scatter_type(type),
	norm_wc(NULL), norm_wc_len(0), norm_wcw(NULL), norm_wcw_len(0), rho(0), m_num_classes(0)
{
}

ScatterSVM::ScatterSVM(float64_t C, std::shared_ptr<Kernel> k, std::shared_ptr<Labels> lab)
: MulticlassSVM(std::make_shared<MulticlassOneVsRestStrategy>(), C, k, lab), scatter_type(NO_BIAS_LIBSVM),
	norm_wc(NULL), norm_wc_len(0), norm_wcw(NULL), norm_wcw_len(0), rho(0), m_num_classes(0)
{
}

ScatterSVM::~ScatterSVM()
{
	SG_FREE(norm_wc);
	SG_FREE(norm_wcw);
}

void ScatterSVM::register_params()
{
	/*m_parameters->add_vector(&norm_wc, &norm_wc_len, "norm_wc", "Norm of w_c");*/
	watch_param("norm_wc", &norm_wc, &norm_wc_len);

	/*m_parameters->add_vector(
	    &norm_wcw, &norm_wcw_len, "norm_wcw", "Norm of w_cw");*/
	watch_param("norm_wcw", &norm_wcw, &norm_wcw_len);

	SG_ADD(&rho, "rho", "Scatter SVM rho");
	SG_ADD(&m_num_classes, "m_num_classes", "Number of classes");

#ifdef USE_SVMLIGHT
	SG_ADD_OPTIONS(
	    (machine_int_t*)&scatter_type, "scatter_type", "Type of scatter SVM",
	    ParameterProperties::NONE,
	    SG_OPTIONS(NO_BIAS_LIBSVM, NO_BIAS_SVMLIGHT, TEST_RULE1, TEST_RULE2));
#else
	SG_ADD_OPTIONS(
	    (machine_int_t*)&scatter_type, "scatter_type", "Type of scatter SVM",
	    ParameterProperties::NONE,
	    SG_OPTIONS(NO_BIAS_LIBSVM, TEST_RULE1, TEST_RULE2));
#endif // USE_SVMLIGHT
}

bool ScatterSVM::train_machine(std::shared_ptr<Features> data)
{
	ASSERT(m_labels && m_labels->get_num_labels())
	ASSERT(m_labels->get_label_type() == LT_MULTICLASS)
	init_strategy();

	m_num_classes = m_multiclass_strategy->get_num_classes();
	int32_t num_vectors = m_labels->get_num_labels();

	if (data)
	{
		if (m_labels->get_num_labels() != data->get_num_vectors())
			SG_ERROR("Number of training vectors does not match number of labels\n")
		m_kernel->init(data, data);
	}

	int32_t* numc=SG_MALLOC(int32_t, m_num_classes);
	SGVector<int32_t>::fill_vector(numc, m_num_classes, 0);

	auto mc = multiclass_labels(m_labels);
	for (int32_t i=0; i<num_vectors; i++)
		numc[(int32_t) mc->get_int_label(i)]++;

	int32_t Nc=0;
	int32_t Nmin=num_vectors;
	for (int32_t i=0; i<m_num_classes; i++)
	{
		if (numc[i]>0)
		{
			Nc++;
			Nmin=Math::min(Nmin, numc[i]);
		}

	}
	SG_FREE(numc);
	m_num_classes=Nc;

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

		SG_INFO("valid nu interval [%f ... %f]\n", nu_min, nu_max)

		if (get_nu()<nu_min || get_nu()>nu_max)
			SG_ERROR("nu out of valid range [%f ... %f]\n", nu_min, nu_max)

		result=train_testrule12();
	}
	else
		SG_ERROR("Unknown Scatter type\n")

	return result;
}

bool ScatterSVM::train_no_bias_libsvm()
{
	svm_problem problem;
	svm_parameter param;
	struct svm_model* model = nullptr;

	struct svm_node* x_space;

	problem.l=m_labels->get_num_labels();
	SG_INFO("%d trainlabels\n", problem.l)

	problem.y=SG_MALLOC(float64_t, problem.l);
	problem.x=SG_MALLOC(struct svm_node*, problem.l);
	x_space=SG_MALLOC(struct svm_node, 2*problem.l);

	for (int32_t i=0; i<problem.l; i++)
	{
		problem.y[i]=+1;
		problem.x[i]=&x_space[2*i];
		x_space[2*i].index=i;
		x_space[2*i+1].index=-1;
	}

	int32_t weights_label[2]={-1,+1};
	float64_t weights[2]={1.0,get_C()/get_C()};

	ASSERT(m_kernel && m_kernel->has_features())
    ASSERT(m_kernel->get_num_vec_lhs()==problem.l)

	param.svm_type=C_SVC; // Nu MC SVM
	param.kernel_type = LINEAR;
	param.degree = 3;
	param.gamma = 0;	// 1/k
	param.coef0 = 0;
	param.nu = get_nu(); // Nu
	auto prev_normalizer=m_kernel->get_normalizer();
	m_kernel->set_normalizer(std::make_shared<ScatterKernelNormalizer>(
				m_num_classes-1, -1, m_labels, prev_normalizer));
	param.kernel=m_kernel.get();
	param.cache_size = m_kernel->get_cache_size();
	param.C = 0;
	param.eps = get_epsilon();
	param.p = 0.1;
	param.shrinking = 0;
	param.nr_weight = 2;
	param.weight_label = weights_label;
	param.weight = weights;
	param.nr_class=m_num_classes;
	param.use_bias = svm_proto()->get_bias_enabled();

	const char* error_msg = svm_check_parameter(&problem,&param);

	if(error_msg)
		SG_ERROR("Error: %s\n",error_msg)

	model = svm_train(&problem, &param);
	m_kernel->set_normalizer(prev_normalizer);

	if (model)
	{
		ASSERT((model->l==0) || (model->l>0 && model->SV && model->sv_coef && model->sv_coef))

		ASSERT(model->nr_class==m_num_classes)
		create_multiclass_svm(m_num_classes);

		rho=model->rho[0];

		SG_FREE(norm_wcw);
		norm_wcw_len = m_machines.size();
		norm_wcw = SG_MALLOC(float64_t, norm_wcw_len);

		for (int32_t i=0; i<m_num_classes; i++)
		{
			int32_t num_sv=model->nSV[i];

			auto svm=std::make_shared<SVM>(num_sv);
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
bool ScatterSVM::train_no_bias_svmlight()
{
	auto prev_normalizer=m_kernel->get_normalizer();
	auto n=std::make_shared<ScatterKernelNormalizer>(
				 m_num_classes-1, -1, m_labels, prev_normalizer);
	m_kernel->set_normalizer(n);
	m_kernel->init_normalizer();

	auto light=std::make_shared<SVMLightOneClass>(get_C(), m_kernel);
	light->set_linadd_enabled(false);
	light->train();

	SG_FREE(norm_wcw);
	norm_wcw = SG_MALLOC(float64_t, m_num_classes);
	norm_wcw_len = m_num_classes;

	int32_t num_sv=light->get_num_support_vectors();
	svm_proto()->create_new_model(num_sv);

	for (int32_t i=0; i<num_sv; i++)
	{
		svm_proto()->set_alpha(i, light->get_alpha(i));
		svm_proto()->set_support_vector(i, light->get_support_vector(i));
	}

	m_kernel->set_normalizer(prev_normalizer);
	return true;
}
#endif //USE_SVMLIGHT

bool ScatterSVM::train_testrule12()
{
	svm_problem problem;
	svm_parameter param;
	struct svm_model* model = nullptr;

	struct svm_node* x_space;
	problem.l=m_labels->get_num_labels();
	SG_INFO("%d trainlabels\n", problem.l)

	problem.y=SG_MALLOC(float64_t, problem.l);
	problem.x=SG_MALLOC(struct svm_node*, problem.l);
	x_space=SG_MALLOC(struct svm_node, 2*problem.l);

	auto mc = multiclass_labels(m_labels);
	for (int32_t i=0; i<problem.l; i++)
	{
		problem.y[i]=mc->get_label(i);
		problem.x[i]=&x_space[2*i];
		x_space[2*i].index=i;
		x_space[2*i+1].index=-1;
	}

	int32_t weights_label[2]={-1,+1};
	float64_t weights[2]={1.0,get_C()/get_C()};

	ASSERT(m_kernel && m_kernel->has_features())
    ASSERT(m_kernel->get_num_vec_lhs()==problem.l)

	param.svm_type=NU_MULTICLASS_SVC; // Nu MC SVM
	param.kernel_type = LINEAR;
	param.degree = 3;
	param.gamma = 0;	// 1/k
	param.coef0 = 0;
	param.nu = get_nu(); // Nu
	param.kernel=m_kernel.get();
	param.cache_size = m_kernel->get_cache_size();
	param.C = 0;
	param.eps = get_epsilon();
	param.p = 0.1;
	param.shrinking = 0;
	param.nr_weight = 2;
	param.weight_label = weights_label;
	param.weight = weights;
	param.nr_class=m_num_classes;
	param.use_bias = svm_proto()->get_bias_enabled();

	const char* error_msg = svm_check_parameter(&problem,&param);

	if(error_msg)
		SG_ERROR("Error: %s\n",error_msg)

	model = svm_train(&problem, &param);

	if (model)
	{
		ASSERT((model->l==0) || (model->l>0 && model->SV && model->sv_coef && model->sv_coef))

		ASSERT(model->nr_class==m_num_classes)
		create_multiclass_svm(m_num_classes);

		rho=model->rho[0];

		SG_FREE(norm_wcw);
		norm_wcw_len = m_machines.size();
		norm_wcw = SG_MALLOC(float64_t, norm_wcw_len);

		for (int32_t i=0; i<m_num_classes; i++)
		{
			int32_t num_sv=model->nSV[i];

			auto svm=std::make_shared<SVM>(num_sv);
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

void ScatterSVM::compute_norm_wc()
{
	SG_FREE(norm_wc);
	norm_wc_len = m_machines.size();
	norm_wc = SG_CALLOC(float64_t, norm_wc_len);
	for (size_t c=0; c<m_machines.size(); c++)
	{
		auto svm=get_svm(c);
		int32_t num_sv = svm->get_num_support_vectors();

		for (int32_t i=0; i<num_sv; i++)
		{
			int32_t ii=svm->get_support_vector(i);
			for (int32_t j=0; j<num_sv; j++)
			{
				int32_t jj=svm->get_support_vector(j);
				norm_wc[c]+=svm->get_alpha(i)*m_kernel->kernel(ii,jj)*svm->get_alpha(j);
			}
		}
	}

	for (size_t i=0; i<m_machines.size(); i++)
		norm_wc[i] = std::sqrt(norm_wc[i]);
}

std::shared_ptr<Labels> ScatterSVM::classify_one_vs_rest()
{
	if (!m_kernel)
	{
		SG_ERROR("SVM can not proceed without kernel!\n")
		return NULL;
	}

	if (!( m_kernel && m_kernel->get_num_vec_lhs() && m_kernel->get_num_vec_rhs()))
		return NULL;

	int32_t num_vectors=m_kernel->get_num_vec_rhs();

	auto output=std::make_shared<MulticlassLabels>(num_vectors);


	if (scatter_type == TEST_RULE1)
	{
		ASSERT(!m_machines.empty())
		for (int32_t i=0; i<num_vectors; i++)
			output->set_label(i, apply_one(i));
	}
#ifdef USE_SVMLIGHT
	else if (scatter_type == NO_BIAS_SVMLIGHT)
	{
		float64_t* outputs=SG_MALLOC(float64_t, num_vectors*m_num_classes);
		SGVector<float64_t>::fill_vector(outputs,num_vectors*m_num_classes,0.0);

		auto mc = multiclass_labels(m_labels);
		for (int32_t i=0; i<num_vectors; i++)
		{
			for (int32_t j=0; j<svm_proto()->get_num_support_vectors(); j++)
			{
				float64_t score=m_kernel->kernel(svm_proto()->get_support_vector(j), i)*svm_proto()->get_alpha(j);
				int32_t label=mc->get_int_label(svm_proto()->get_support_vector(j));
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
		ASSERT(!m_machines.empty())
		ASSERT(num_vectors==output->get_num_labels())
		std::vector<std::shared_ptr<Labels>> outputs(m_machines.size());

		for (size_t i=0; i<m_machines.size(); i++)
		{
			//SG_PRINT("svm %d\n", i)
			auto svm = get_svm(i);
			ASSERT(svm)
			svm->set_kernel(m_kernel);
			svm->set_labels(m_labels);
			outputs[i]=svm->apply();

		}

		for (int32_t i=0; i<num_vectors; i++)
		{
			int32_t winner=0;
			float64_t max_out=outputs[0]->as<RegressionLabels>()->get_label(i)/norm_wc[0];

			for (size_t j=1; j<m_machines.size(); j++)
			{
				float64_t out=outputs[j]->as<RegressionLabels>()->get_label(i)/norm_wc[j];

				if (out>max_out)
				{
					winner=j;
					max_out=out;
				}
			}

			output->set_label(i, winner);
		}

	}

	return output;
}

float64_t ScatterSVM::apply_one(int32_t num)
{
	ASSERT(!m_machines.empty())
	float64_t* outputs=SG_MALLOC(float64_t, m_machines.size());
	int32_t winner=0;

	if (scatter_type == TEST_RULE1)
	{
		for (size_t c=0; c<m_machines.size(); c++)
			outputs[c]=get_svm(c)->get_bias()-rho;

		for (size_t c=0; c<m_machines.size(); c++)
		{
			float64_t v=0;

			for (int32_t i=0; i<get_svm(c)->get_num_support_vectors(); i++)
			{
				float64_t alpha=get_svm(c)->get_alpha(i);
				int32_t svidx=get_svm(c)->get_support_vector(i);
				v += alpha*m_kernel->kernel(svidx, num);
			}

			outputs[c] += v;
			for (size_t j=0; j<m_machines.size(); j++)
				outputs[j] -= v/m_machines.size();
		}

		for (size_t j=0; j<m_machines.size(); j++)
			outputs[j]/=norm_wcw[j];

		float64_t max_out=outputs[0];
		for (size_t j=0; j<m_machines.size(); j++)
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
		SG_ERROR("Use classify...\n")
	}
#endif //USE_SVMLIGHT
	else
	{
		float64_t max_out=get_svm(0)->apply_one(num)/norm_wc[0];

		for (size_t i=1; i<m_machines.size(); i++)
		{
			outputs[i]=get_svm(i)->apply_one(num)/norm_wc[i];
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
