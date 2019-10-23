/*

 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 1999-2009 Soeren Sonnenburg
 * Written (W) 2013 Heiko Strathmann
 * Copyright (C) 1999-2009 Fraunhofer Institute FIRST and Max-Planck-Society
 */

#include <shogun/regression/svr/LibSVR.h>
#include <shogun/labels/RegressionLabels.h>
#include <shogun/io/SGIO.h>

#include <utility>

using namespace shogun;

LibSVR::LibSVR()
: SVM()
{
	register_params();
	solver_type=LIBSVR_EPSILON_SVR;
}

LibSVR::LibSVR(float64_t C, float64_t svr_param, std::shared_ptr<Kernel> k, std::shared_ptr<Labels> lab,
		LIBSVR_SOLVER_TYPE st)
: SVM()
{
	register_params();
	set_C(C,C);

	switch (st)
	{
	case LIBSVR_EPSILON_SVR:
		set_tube_epsilon(svr_param);
		break;
	case LIBSVR_NU_SVR:
		set_nu(svr_param);
		break;
	default:
		error("LibSVR::LibSVR(): Unknown solver type!");
		break;
	}

	set_labels(std::move(lab));
	set_kernel(std::move(k));
	solver_type=st;
}

LibSVR::~LibSVR()
{
}

void LibSVR::register_params()
{
	SG_ADD_OPTIONS(
	    (machine_int_t*)&solver_type, "libsvr_solver_type",
	    "LibSVR Solver type", ParameterProperties::NONE,
	    SG_OPTIONS(LIBSVR_EPSILON_SVR, LIBSVR_NU_SVR));
}

EMachineType LibSVR::get_classifier_type()
{
	return CT_LIBSVR;
}

bool LibSVR::train_machine(std::shared_ptr<Features> data)
{
	svm_problem problem;
	svm_parameter param;
	struct svm_model* model = nullptr;

	ASSERT(kernel)
	ASSERT(m_labels && m_labels->get_num_labels())

	if (data)
	{
		if (m_labels->get_num_labels() != data->get_num_vectors())
			error("Number of training vectors does not match number of labels");
		kernel->init(data, data);
	}

	SG_FREE(model);

	struct svm_node* x_space;

	problem.l=m_labels->get_num_labels();
	io::info("{} trainlabels", problem.l);

	problem.y=SG_MALLOC(float64_t, problem.l);
	problem.x=SG_MALLOC(struct svm_node*, problem.l);
	x_space=SG_MALLOC(struct svm_node, 2*problem.l);

	auto labels = regression_labels(m_labels);
	for (int32_t i=0; i<problem.l; i++)
	{
		problem.y[i] = labels->get_label(i);
		problem.x[i]=&x_space[2*i];
		x_space[2*i].index=i;
		x_space[2*i+1].index=-1;
	}

	int32_t weights_label[2]={-1,+1};
	float64_t weights[2]={1.0,get_C2()/get_C1()};

	switch (solver_type)
	{
	case LIBSVR_EPSILON_SVR:
		param.svm_type=EPSILON_SVR;
		break;
	case LIBSVR_NU_SVR:
		param.svm_type=NU_SVR;
		break;
	default:
		error("{}::train_machine(): Unknown solver type!", get_name());
		break;
	}

	param.kernel_type = LINEAR;
	param.degree = 3;
	param.gamma = 0;	// 1/k
	param.coef0 = 0;
	param.nu = nu;
	param.kernel=kernel.get();
	param.cache_size = kernel->get_cache_size();
	param.max_train_time = m_max_train_time;
	param.C = get_C1();
	param.eps = epsilon;
	param.p = tube_epsilon;
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
		SG_FREE(problem.y);
		SG_FREE(x_space);

		svm_destroy_model(model);
		model=NULL;
		return true;
	}
	else
		return false;
}
