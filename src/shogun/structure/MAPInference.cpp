/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2013 Shell Hu
 * Copyright (C) 2013 Shell Hu
 */

#include <shogun/structure/MAPInference.h>
#include <shogun/structure/BeliefPropagation.h>
#include <shogun/structure/GraphCut.h>
#include <shogun/structure/GEMPLP.h>
#include <shogun/labels/FactorGraphLabels.h>

using namespace shogun;

CMAPInference::CMAPInference() : CSGObject()
{
	SG_UNSTABLE("CMAPInference::CMAPInference()", "\n");

	init();
}

CMAPInference::CMAPInference(CFactorGraph* fg, EMAPInferType inference_method)
	: CSGObject()
{
	init();
	m_fg = fg;

	REQUIRE(fg != NULL, "fg cannot be NULL!\n");

	switch(inference_method)
	{
		case TREE_MAX_PROD:
			m_infer_impl = new CTreeMaxProduct(fg);
			break;
		case GRAPH_CUT:
			m_infer_impl = new CGraphCut(fg);
			break;
		case GEMPLP:
			m_infer_impl = new CGEMPLP(fg);
			break;
		case LOOPY_MAX_PROD:
			SG_ERROR("LoopyMaxProduct has not been implemented!\n");
			break;
		case LP_RELAXATION:
			SG_ERROR("LPRelaxation has not been implemented!\n");
			break;
		case TRWS_MAX_PROD:
			SG_ERROR("TRW-S has not been implemented!\n");
			break;
		default:
			SG_ERROR("Unsupported inference method!\n");
			break;
	}

	SG_REF(m_infer_impl);
	SG_REF(m_fg);
}

CMAPInference::~CMAPInference()
{
	SG_UNREF(m_infer_impl);
	SG_UNREF(m_outputs);
	SG_UNREF(m_fg);
}

void CMAPInference::init()
{
	SG_ADD((CSGObject**)&m_fg, "fg", "factor graph", MS_NOT_AVAILABLE);
	SG_ADD((CSGObject**)&m_outputs, "outputs", "Structured outputs", MS_NOT_AVAILABLE);
	SG_ADD((CSGObject**)&m_infer_impl, "infer_impl", "Inference implementation", MS_NOT_AVAILABLE);
	SG_ADD(&m_energy, "energy", "Minimized energy", MS_NOT_AVAILABLE);

	m_outputs = NULL;
	m_infer_impl = NULL;
	m_fg = NULL;
	m_energy = 0;
}

void CMAPInference::inference()
{
	SGVector<int32_t> assignment(m_fg->get_num_vars());
	assignment.zero();
	m_energy = m_infer_impl->inference(assignment);

	// create structured output, with default normalized hamming loss
	SG_UNREF(m_outputs);
	SGVector<float64_t> loss_weights(m_fg->get_num_vars());
	SGVector<float64_t>::fill_vector(loss_weights.vector, loss_weights.vlen, 1.0 / loss_weights.vlen);
	m_outputs = new CFactorGraphObservation(assignment, loss_weights); // already ref() in constructor
	SG_REF(m_outputs);
}

CFactorGraphObservation* CMAPInference::get_structured_outputs() const
{
	SG_REF(m_outputs);
	return m_outputs;
}

float64_t CMAPInference::get_energy() const
{
	return m_energy;
}

//-----------------------------------------------------------------

CMAPInferImpl::CMAPInferImpl() : CSGObject()
{
	register_parameters();
}

CMAPInferImpl::CMAPInferImpl(CFactorGraph* fg)
	: CSGObject()
{
	register_parameters();
	m_fg = fg;
}

CMAPInferImpl::~CMAPInferImpl()
{
}

void CMAPInferImpl::register_parameters()
{
	SG_ADD((CSGObject**)&m_fg, "fg",
		"Factor graph pointer", MS_NOT_AVAILABLE);

	m_fg = NULL;
}

