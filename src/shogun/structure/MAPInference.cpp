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

#include <string>

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

	REQUIRE(fg != NULL, "%s::CMAPInference(): fg cannot be NULL!\n", get_name());

	switch(inference_method)
	{
		case TREE_MAX_PROD:
			m_infer_impl = new CTreeMaxProduct(fg);
			break;
		case LOOPY_MAX_PROD:
			SG_ERROR("%s::CMAPInference(): LoopyMaxProduct has not been implemented!\n", 
				get_name());
			break;
		case LP_RELAXATION:
			SG_ERROR("%s::CMAPInference(): LPRelaxation has not been implemented!\n", 
				get_name());
			break;
		case TRWS_MAX_PROD:
			SG_ERROR("%s::CMAPInference(): TRW-S has not been implemented!\n", 
				get_name());
			break;
		case ITER_COND_MODE:
			SG_ERROR("%s::CMAPInference(): ICM has not been implemented!\n", 
				get_name());
			break;
		case NAIVE_MEAN_FIELD:
			SG_ERROR("%s::CMAPInference(): NaiveMeanField has not been implemented!\n", 
				get_name());
			break;
		case STRUCT_MEAN_FIELD:
			SG_ERROR("%s::CMAPInference(): StructMeanField has not been implemented!\n", 
				get_name());
			break;
		default:
			SG_ERROR("%s::CMAPInference(): unsupported inference method!\n", 
				get_name());
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

	SG_UNREF(m_outputs);
	m_outputs = new CFactorGraphObservation(assignment); // already ref() in constructor
	SG_REF(m_outputs);
}

void CMAPInference::loss_augmentation(CFactorGraphObservation* gt)
{
	loss_augmentation(gt->get_data(), gt->get_loss_weights());
}

void CMAPInference::loss_augmentation(SGVector<int32_t> states_gt, SGVector<float64_t> loss)
{
	if (loss.size() == 0)
	{
		loss.resize_vector(states_gt.size());
		SGVector<float64_t>::fill_vector(loss.vector, loss.vlen, 1.0);	
	}

	int32_t num_vars = states_gt.size();
	ASSERT(num_vars == loss.size());

	SGVector<int32_t> var_flags(num_vars);
	var_flags.zero();

	// augment loss to incorrect states in the first factor containing the variable
	// since += L_i for each variable if it takes wrong state ever
	// TODO: augment unary factors 
	CDynamicObjectArray* facs = m_fg->get_factors();
	for (int32_t fi = 0; fi < facs->get_num_elements(); ++fi)
	{
		CFactor* fac = dynamic_cast<CFactor*>(facs->get_element(fi));
		SGVector<int32_t> vars = fac->get_variables();
		for (int32_t vi = 0; vi < vars.size(); vi++)
		{
			int32_t vv = vars[vi];
			ASSERT(vv < num_vars);
			if (var_flags[vv])
				continue;

			SGVector<float64_t> energies = fac->get_energies();
			for (int32_t ei = 0; ei < energies.size(); ei++)
			{
				CTableFactorType* ftype = fac->get_factor_type();
				int32_t vstate = ftype->state_from_index(ei, vi);
				SG_UNREF(ftype);

				if (states_gt[vv] == vstate)
					continue;

				// -delta(y_n, y_i_n)
				fac->set_energy(ei, energies[ei] - loss[vv]);
			}

			var_flags[vv] = 1;
		}

		SG_UNREF(fac);
	}

	SG_UNREF(facs);

	// make sure all variables have been checked
	int32_t min_var = SGVector<int32_t>::min(var_flags.vector, var_flags.vlen);
	ASSERT(min_var == 1);
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

