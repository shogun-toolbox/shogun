/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2013 Thoralf Klein
 * Written (W) 2012 Fernando José Iglesias García
 * Copyright (C) 2012 Fernando José Iglesias García
 */

#include <shogun/machine/StructuredOutputMachine.h>

using namespace shogun;

CStructuredOutputMachine::CStructuredOutputMachine()
: CMachine(), m_model(NULL), m_surrogate_loss(NULL)
{
	register_parameters();
}

CStructuredOutputMachine::CStructuredOutputMachine(
		CStructuredModel*  model,
		CStructuredLabels* labs)
: CMachine(), m_model(model), m_surrogate_loss(NULL)
{
	SG_REF(m_model);
	set_labels(labs);
	register_parameters();
}

CStructuredOutputMachine::~CStructuredOutputMachine()
{
	SG_UNREF(m_model);
	SG_UNREF(m_surrogate_loss);
}

void CStructuredOutputMachine::set_model(CStructuredModel* model)
{
	SG_REF(model);
	SG_UNREF(m_model);
	m_model = model;
}

CStructuredModel* CStructuredOutputMachine::get_model() const
{
	SG_REF(m_model);
	return m_model;
}

void CStructuredOutputMachine::register_parameters()
{
	SG_ADD((CSGObject**)&m_model, "m_model", "Structured model", MS_NOT_AVAILABLE);
	SG_ADD((CSGObject**)&m_surrogate_loss, "m_surrogate_loss", "Surrogate loss", MS_NOT_AVAILABLE);
}

void CStructuredOutputMachine::set_labels(CLabels* lab)
{
	CMachine::set_labels(lab);
	REQUIRE(m_model != NULL, "please call set_model() before set_labels()\n");
	m_model->set_labels(CLabelsFactory::to_structured(lab));
}

void CStructuredOutputMachine::set_features(CFeatures* f)
{
	m_model->set_features(f);
}

CFeatures* CStructuredOutputMachine::get_features() const
{
	return m_model->get_features();
}

void CStructuredOutputMachine::set_surrogate_loss(CLossFunction* loss)
{
	SG_REF(loss);
	SG_UNREF(m_surrogate_loss);
	m_surrogate_loss = loss;
}

CLossFunction* CStructuredOutputMachine::get_surrogate_loss() const
{
	SG_REF(m_surrogate_loss);
	return m_surrogate_loss;
}

float64_t CStructuredOutputMachine::risk_nslack_margin_rescale(float64_t* subgrad, float64_t* W, TMultipleCPinfo* info)
{
	int32_t dim = m_model->get_dim();
	
	int32_t from=0, to=0;
	CFeatures* features = get_features();
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
	SG_UNREF(features);

	float64_t R = 0.0;
	for (int32_t i=0; i<dim; i++)
		subgrad[i] = 0;

	for (int32_t i=from; i<to; i++)
	{
		CResultSet* result = m_model->argmax(SGVector<float64_t>(W,dim,false), i, true);
		SGVector<float64_t> psi_pred = result->psi_pred;
		SGVector<float64_t> psi_truth = result->psi_truth;
		SGVector<float64_t>::vec1_plus_scalar_times_vec2(subgrad, 1.0, psi_pred.vector, dim);
		SGVector<float64_t>::vec1_plus_scalar_times_vec2(subgrad, -1.0, psi_truth.vector, dim);
		R += result->score;
		SG_UNREF(result);
	}

	return R;
}

float64_t CStructuredOutputMachine::risk_nslack_slack_rescale(float64_t* subgrad, float64_t* W, TMultipleCPinfo* info)
{
	SG_ERROR("%s::risk_nslack_slack_rescale() has not been implemented!\n", get_name());
	return 0.0;
}

float64_t CStructuredOutputMachine::risk_1slack_margin_rescale(float64_t* subgrad, float64_t* W, TMultipleCPinfo* info)
{
	SG_ERROR("%s::risk_1slack_margin_rescale() has not been implemented!\n", get_name());
	return 0.0;
}

float64_t CStructuredOutputMachine::risk_1slack_slack_rescale(float64_t* subgrad, float64_t* W, TMultipleCPinfo* info)
{
	SG_ERROR("%s::risk_1slack_slack_rescale() has not been implemented!\n", get_name());
	return 0.0;
}

float64_t CStructuredOutputMachine::risk_customized_formulation(float64_t* subgrad, float64_t* W, TMultipleCPinfo* info)
{
	SG_ERROR("%s::risk_customized_formulation() has not been implemented!\n", get_name());
	return 0.0;
}

float64_t CStructuredOutputMachine::risk(float64_t* subgrad, float64_t* W, 
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
			SG_ERROR("%s::risk(): cannot recognize the risk type!\n", get_name());
			ret = -1;
			break;
	}
	return ret;
}
