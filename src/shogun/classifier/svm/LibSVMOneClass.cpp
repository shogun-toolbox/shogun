/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Soeren Sonnenburg, Heiko Strathmann, Sergey Lisitsyn,
 *          Leon Kuchenbecker
 */

#include <shogun/classifier/svm/LibSVMOneClass.h>
#include <shogun/io/SGIO.h>

#include <utility>

using namespace shogun;

LibSVMOneClass::LibSVMOneClass()
: SVM()
{
}

LibSVMOneClass::LibSVMOneClass(float64_t C, std::shared_ptr<Kernel> k)
: SVM(C, std::move(k), NULL)
{
}

LibSVMOneClass::~LibSVMOneClass()
{
}

bool LibSVMOneClass::train_machine(std::shared_ptr<Features> data)
{
	svm_problem problem;
	svm_parameter param;
	struct svm_model* model = nullptr;

	ASSERT(kernel)
	if (data)
		kernel->init(data, data);

	problem.l=kernel->get_num_vec_lhs();

	struct svm_node* x_space;
	io::info("{} train data points", problem.l);

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
	param.kernel=kernel.get();
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
		error("Error: {}",error_msg);
	model = svm_train(&problem, &param);

	if (model)
	{
		ASSERT(model->nr_class==2)
		ASSERT((model->l==0) || (model->l>0 && model->SV && model->sv_coef && model->sv_coef[0]))

		int32_t num_sv=model->l;

		create_new_model(num_sv);
		SVM::set_objective(model->objective);

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
