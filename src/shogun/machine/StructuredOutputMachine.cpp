/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Fernando Iglesias, Thoralf Klein, Shell Hu, Bjoern Esser,
 *          Viktor Gal
 */

#include <shogun/machine/StructuredOutputMachine.h>
#include <shogun/labels/StructuredLabels.h>
#include <shogun/loss/LossFunction.h>
#include <shogun/structure/StructuredModel.h>

#include <shogun/mathematics/linalg/LinalgNamespace.h>

#include <utility>

using namespace shogun;

StructuredOutputMachine::StructuredOutputMachine()
: Machine(), m_model(NULL), m_surrogate_loss(NULL)
{
	register_parameters();
}

StructuredOutputMachine::StructuredOutputMachine(
		std::shared_ptr<StructuredModel>  model,
		const std::shared_ptr<StructuredLabels>& labs)
: Machine(), m_model(std::move(model)), m_surrogate_loss(NULL)
{

	set_labels(labs);
	register_parameters();
}

StructuredOutputMachine::~StructuredOutputMachine()
{



}

void StructuredOutputMachine::set_model(std::shared_ptr<StructuredModel> model)
{


	m_model = std::move(model);
}

std::shared_ptr<StructuredModel> StructuredOutputMachine::get_model() const
{

	return m_model;
}

void StructuredOutputMachine::register_parameters()
{
	SG_ADD((std::shared_ptr<SGObject>*)&m_model, "m_model", "Structured model");
	SG_ADD(&m_surrogate_loss, "m_surrogate_loss", "Surrogate loss");
	SG_ADD(&m_verbose, "verbose", "Verbosity flag");
	SG_ADD((std::shared_ptr<SGObject>*)&m_helper, "helper", "Training helper");

	m_verbose = false;
	m_helper = NULL;
}

void StructuredOutputMachine::set_labels(std::shared_ptr<Labels> lab)
{
	Machine::set_labels(lab);
	require(m_model != NULL, "please call set_model() before set_labels()");
	m_model->set_labels(lab->as<StructuredLabels>());
}

void StructuredOutputMachine::set_features(std::shared_ptr<Features> f)
{
	m_model->set_features(std::move(f));
}

std::shared_ptr<Features> StructuredOutputMachine::get_features() const
{
	return m_model->get_features();
}

void StructuredOutputMachine::set_surrogate_loss(std::shared_ptr<LossFunction> loss)
{


	m_surrogate_loss = std::move(loss);
}

std::shared_ptr<LossFunction> StructuredOutputMachine::get_surrogate_loss() const
{

	return m_surrogate_loss;
}

float64_t StructuredOutputMachine::risk_nslack_margin_rescale(SGVector<float64_t>& subgrad, SGVector<float64_t>& W, TMultipleCPinfo* info)
{
	int32_t dim = m_model->get_dim();

	int32_t from=0, to=0;
	auto features = get_features();
	if (info)
	{
		from = info->m_from;
		to = (info->m_N == 0) ? features->get_num_vectors() : from+info->m_N;
	}
	else
	{
		from = 0;
		to = features->get_num_vectors();
	}


	float64_t R = 0.0;
	linalg::zero(subgrad);

	for (int32_t i=from; i<to; i++)
	{
		auto result = m_model->argmax(SGVector<float64_t>(W.vector,dim,false), i, true);
		SGVector<float64_t> psi_pred = result->psi_pred;
		SGVector<float64_t> psi_truth = result->psi_truth;
		SGVector<float64_t>::vec1_plus_scalar_times_vec2(subgrad.vector, 1.0, psi_pred.vector, dim);
		SGVector<float64_t>::vec1_plus_scalar_times_vec2(subgrad.vector, -1.0, psi_truth.vector, dim);
		R += result->score;

	}

	return R;
}

float64_t StructuredOutputMachine::risk_nslack_slack_rescale(SGVector<float64_t>& subgrad, SGVector<float64_t>& W, TMultipleCPinfo* info)
{
	error("{}::risk_nslack_slack_rescale() has not been implemented!", get_name());
	return 0.0;
}

float64_t StructuredOutputMachine::risk_1slack_margin_rescale(SGVector<float64_t>& subgrad, SGVector<float64_t>& W, TMultipleCPinfo* info)
{
	error("{}::risk_1slack_margin_rescale() has not been implemented!", get_name());
	return 0.0;
}

float64_t StructuredOutputMachine::risk_1slack_slack_rescale(SGVector<float64_t>& subgrad, SGVector<float64_t>& W, TMultipleCPinfo* info)
{
	error("{}::risk_1slack_slack_rescale() has not been implemented!", get_name());
	return 0.0;
}

float64_t StructuredOutputMachine::risk_customized_formulation(SGVector<float64_t>& subgrad, SGVector<float64_t>& W, TMultipleCPinfo* info)
{
	error("{}::risk_customized_formulation() has not been implemented!", get_name());
	return 0.0;
}

float64_t StructuredOutputMachine::risk(SGVector<float64_t>& subgrad, SGVector<float64_t>& W,
		TMultipleCPinfo* info, EStructRiskType rtype)
{
	float64_t ret = 0.0;
	switch(rtype)
	{
		case N_SLACK_MARGIN_RESCALING:
			ret = risk_nslack_margin_rescale(subgrad, W, info);
			break;
		case N_SLACK_SLACK_RESCALING:
			ret = risk_nslack_slack_rescale(subgrad, W, info);
			break;
		case ONE_SLACK_MARGIN_RESCALING:
			ret = risk_1slack_margin_rescale(subgrad, W, info);
			break;
		case ONE_SLACK_SLACK_RESCALING:
			ret = risk_1slack_slack_rescale(subgrad, W, info);
			break;
		case CUSTOMIZED_RISK:
			ret = risk_customized_formulation(subgrad, W, info);
			break;
		default:
			error("{}::risk(): cannot recognize the risk type!", get_name());
			ret = -1;
			break;
	}
	return ret;
}

std::shared_ptr<SOSVMHelper> StructuredOutputMachine::get_helper() const
{
	if (m_helper == NULL)
	{
		error("{}::get_helper(): no helper has been created!"
			"Please set verbose before training!", get_name());
	}


	return m_helper;
}

void StructuredOutputMachine::set_verbose(bool verbose)
{
	m_verbose = verbose;
}

bool StructuredOutputMachine::get_verbose() const
{
	return m_verbose;
}
