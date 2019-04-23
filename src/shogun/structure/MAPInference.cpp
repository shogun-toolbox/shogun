/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Shell Hu, Jiaolong Xu, Bjoern Esser
 */

#include <shogun/structure/MAPInference.h>
#include <shogun/structure/BeliefPropagation.h>
#include <shogun/structure/GraphCut.h>
#include <shogun/structure/GEMPLP.h>
#include <shogun/labels/FactorGraphLabels.h>

using namespace shogun;

MAPInference::MAPInference() : SGObject()
{
	unstable(SOURCE_LOCATION);

	init();
}

MAPInference::MAPInference(std::shared_ptr<FactorGraph> fg, EMAPInferType inference_method)
	: SGObject()
{
	init();
	m_fg = fg;

	require(fg != NULL, "{}::MAPInference(): fg cannot be NULL!", get_name());

	switch(inference_method)
	{
		case TREE_MAX_PROD:
			m_infer_impl = std::make_shared<TreeMaxProduct>(fg);
			break;
		case GRAPH_CUT:
			m_infer_impl = std::make_shared<GraphCut>(fg);
			break;
		case GEMP_LP:
			m_infer_impl = std::make_shared<GEMPLP>(fg);
			break;
		case LOOPY_MAX_PROD:
			error("{}::MAPInference(): LoopyMaxProduct has not been implemented!",
				get_name());
			break;
		case LP_RELAXATION:
			error("{}::MAPInference(): LPRelaxation has not been implemented!",
				get_name());
			break;
		case TRWS_MAX_PROD:
			error("{}::MAPInference(): TRW-S has not been implemented!",
				get_name());
			break;
		default:
			error("{}::CMAPInference(): unsupported inference method!",
				get_name());
			break;
	}



}

MAPInference::~MAPInference()
{
}

void MAPInference::init()
{
	SG_ADD((std::shared_ptr<SGObject>*)&m_fg, "fg", "factor graph");
	SG_ADD((std::shared_ptr<SGObject>*)&m_outputs, "outputs", "Structured outputs");
	SG_ADD((std::shared_ptr<SGObject>*)&m_infer_impl, "infer_impl", "Inference implementation");
	SG_ADD(&m_energy, "energy", "Minimized energy");

	m_outputs = NULL;
	m_infer_impl = NULL;
	m_fg = NULL;
	m_energy = 0;
}

void MAPInference::inference()
{
	SGVector<int32_t> assignment(m_fg->get_num_vars());
	assignment.zero();
	m_energy = m_infer_impl->inference(assignment);

	// create structured output, with default normalized hamming loss

	SGVector<float64_t> loss_weights(m_fg->get_num_vars());
	SGVector<float64_t>::fill_vector(loss_weights.vector, loss_weights.vlen, 1.0 / loss_weights.vlen);
	m_outputs = std::make_shared<FactorGraphObservation>(assignment, loss_weights); // already ref() in constructor

}

std::shared_ptr<FactorGraphObservation> MAPInference::get_structured_outputs() const
{

	return m_outputs;
}

float64_t MAPInference::get_energy() const
{
	return m_energy;
}

//-----------------------------------------------------------------

MAPInferImpl::MAPInferImpl() : SGObject()
{
	register_parameters();
}

MAPInferImpl::MAPInferImpl(std::shared_ptr<FactorGraph> fg)
	: SGObject()
{
	register_parameters();
	m_fg = fg;
}

MAPInferImpl::~MAPInferImpl()
{
}

void MAPInferImpl::register_parameters()
{
	SG_ADD((std::shared_ptr<SGObject>*)&m_fg, "fg",
		"Factor graph pointer");

	m_fg = NULL;
}

