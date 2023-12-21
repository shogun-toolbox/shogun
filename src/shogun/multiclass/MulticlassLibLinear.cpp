/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Sergey Lisitsyn, Soeren Sonnenburg, Chiyuan Zhang, Giovanni De Toni,
 *          Evan Shelhamer
 */

#include <shogun/lib/config.h>
#include <shogun/multiclass/MulticlassLibLinear.h>
#include <shogun/multiclass/MulticlassOneVsRestStrategy.h>
#include <shogun/mathematics/Math.h>
#include <shogun/lib/v_array.h>
#include <shogun/lib/Signal.h>
#include <shogun/labels/MulticlassLabels.h>

#include <utility>

using namespace shogun;

MulticlassLibLinear::MulticlassLibLinear() :
	RandomMixin<LinearMulticlassMachine>()
{
	register_parameters();
	init_defaults();
}

MulticlassLibLinear::MulticlassLibLinear(float64_t C) :
	RandomMixin<LinearMulticlassMachine>(std::make_shared<MulticlassOneVsRestStrategy>(), nullptr)
{
	register_parameters();
	init_defaults();
	set_C(C);
}

void MulticlassLibLinear::init_defaults()
{
	set_C(1.0);
	set_epsilon(1e-2);
	set_max_iter(10000);
	set_use_bias(false);
	set_save_train_state(false);
	m_train_state = NULL;
}

void MulticlassLibLinear::register_parameters()
{
	SG_ADD(&m_C, "C", "regularization constant",ParameterProperties::HYPER);
	SG_ADD(&m_epsilon, "epsilon", "tolerance epsilon");
	SG_ADD(&m_max_iter, "max_iter", "max number of iterations");
	SG_ADD(&m_use_bias, "use_bias", "indicates whether bias should be used");
}

MulticlassLibLinear::~MulticlassLibLinear()
{
	reset_train_state();
}

SGVector<int32_t> MulticlassLibLinear::get_support_vectors() const
{
	if (!m_train_state)
		error("Please enable save_train_state option and train machine.");

	v_array<int32_t> nz_idxs;
	nz_idxs.reserve(m_num_vectors);

	for (int32_t i=0; i<m_num_vectors; i++)
	{
		for (int32_t y=0; y<m_num_classes; y++)
		{
			if (Math::abs(m_train_state->alpha[i*m_num_classes+y])>1e-6)
			{
				nz_idxs.push(i);
				break;
			}
		}
	}
	int32_t num_nz = nz_idxs.index();
	nz_idxs.reserve(num_nz);
	return SGVector<int32_t>(nz_idxs.begin,num_nz);
}

SGMatrix<float64_t> MulticlassLibLinear::obtain_regularizer_matrix() const
{
	return SGMatrix<float64_t>();
}

bool MulticlassLibLinear::train_machine(const std::shared_ptr<Features>& data, const std::shared_ptr<Labels>& labs)
{
	require(m_multiclass_strategy, "Multiclass strategy not set");
	init_strategy(labs);
	auto feats = data->as<DotFeatures>();
	m_num_vectors = data->get_num_vectors();
	m_num_classes = multiclass_labels(labs)->get_num_classes();
	int32_t bias_n = m_use_bias ? 1 : 0;

	liblinear_problem mc_problem;
	mc_problem.l = m_num_vectors;
	mc_problem.n = feats->get_dim_feature_space() + bias_n;
	mc_problem.y = SG_MALLOC(float64_t, mc_problem.l);
	for (int32_t i=0; i<m_num_vectors; i++)
		mc_problem.y[i] = multiclass_labels(labs)->get_int_label(i);

	mc_problem.x = feats;
	mc_problem.use_bias = m_use_bias;

	SGMatrix<float64_t> w0 = obtain_regularizer_matrix();

	if (!m_train_state)
		m_train_state = new mcsvm_state();

	float64_t* C = SG_MALLOC(float64_t, m_num_vectors);
	for (int32_t i=0; i<m_num_vectors; i++)
		C[i] = m_C;

	Solver_MCSVM_CS solver(&mc_problem,m_num_classes,C,w0.matrix,m_epsilon,
	                       m_max_iter,m_max_train_time,m_train_state);
	solver.solve(m_prng);

	m_machines.clear();
	for (int32_t i=0; i<m_num_classes; i++)
	{
		auto machine = std::make_shared<LinearMachine>();
		SGVector<float64_t> cw(mc_problem.n-bias_n);

		for (int32_t j=0; j<mc_problem.n-bias_n; j++)
			cw[j] = m_train_state->w[j*m_num_classes+i];

		machine->set_w(cw);

		if (m_use_bias)
			machine->set_bias(m_train_state->w[(mc_problem.n-bias_n)*m_num_classes+i]);

		m_machines.push_back(machine);
	}

	if (!m_save_train_state)
		reset_train_state();

	SG_FREE(C);
	SG_FREE(mc_problem.y);

	return true;
}
