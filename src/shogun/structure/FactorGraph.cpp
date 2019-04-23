/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Shell Hu, Soeren Sonnenburg, Sanuj Sharma, Fernando Iglesias,
 *          Bjoern Esser
 */

#include <shogun/structure/FactorGraph.h>
#include <shogun/labels/FactorGraphLabels.h>

using namespace shogun;

FactorGraph::FactorGraph()
	: SGObject()
{
	SG_UNSTABLE("FactorGraph::FactorGraph()", "\n");

	register_parameters();
	init();
}

FactorGraph::FactorGraph(SGVector<int32_t> card)
	: SGObject()
{
	m_cardinalities = card;
	register_parameters();
	init();
}

FactorGraph::FactorGraph(const FactorGraph &fg)
	: SGObject()
{
	register_parameters();
	m_cardinalities = fg.get_cardinalities();
	// No need to unref and ref in this case
	m_factors = fg.get_factors();
	m_datasources = fg.get_factor_data_sources();
	m_dset = fg.get_disjoint_set();
	m_has_cycle = !(fg.is_acyclic_graph());
	m_num_edges = fg.get_num_edges();
}

FactorGraph::~FactorGraph()
{

}

void FactorGraph::register_parameters()
{
	SG_ADD(&m_cardinalities, "cardinalities", "Cardinalities");
	SG_ADD((std::shared_ptr<SGObject>*)&m_factors, "factors", "Factors");
	SG_ADD((std::shared_ptr<SGObject>*)&m_datasources, "datasources", "Factor data sources");
	SG_ADD((std::shared_ptr<SGObject>*)&m_dset, "dset", "Disjoint set");
	SG_ADD(&m_has_cycle, "has_cycle", "Whether has circle in graph");
	SG_ADD(&m_num_edges, "num_edges", "Number of edges");
}

void FactorGraph::init()
{
	m_has_cycle = false;
	m_num_edges = 0;
	m_factors = NULL;
	m_datasources = NULL;
	m_factors = std::make_shared<DynamicObjectArray>();
	m_datasources = std::make_shared<DynamicObjectArray>();

	// NOTE m_cards cannot be empty
	m_dset = std::make_shared<DisjointSet>(m_cardinalities.size());




}

void FactorGraph::add_factor(std::shared_ptr<Factor> factor)
{
	m_factors->push_back(factor);
	m_num_edges += factor->get_variables().size();

	// graph structure changed after adding factors
	if (m_dset->get_connected())
		m_dset->set_connected(false);
}

void FactorGraph::add_data_source(std::shared_ptr<FactorDataSource> datasource)
{
	m_datasources->push_back(datasource);
}

std::shared_ptr<DynamicObjectArray> FactorGraph::get_factors() const
{

	return m_factors;
}

std::shared_ptr<DynamicObjectArray> FactorGraph::get_factor_data_sources() const
{

	return m_datasources;
}

int32_t FactorGraph::get_num_factors() const
{
	return m_factors->get_num_elements();
}

SGVector<int32_t> FactorGraph::get_cardinalities() const
{
	return m_cardinalities;
}

void FactorGraph::set_cardinalities(SGVector<int32_t> cards)
{
	m_cardinalities = cards.clone();
}

std::shared_ptr<DisjointSet> FactorGraph::get_disjoint_set() const
{

	return m_dset;
}

int32_t FactorGraph::get_num_edges() const
{
	return m_num_edges;
}

int32_t FactorGraph::get_num_vars() const
{
	return m_cardinalities.size();
}

void FactorGraph::compute_energies()
{
	for (int32_t fi = 0; fi < m_factors->get_num_elements(); ++fi)
	{
		auto fac = m_factors->get_element<Factor>(fi);
		fac->compute_energies();

	}
}

float64_t FactorGraph::evaluate_energy(const SGVector<int32_t> state) const
{
	ASSERT(state.size() == m_cardinalities.size());

	float64_t energy = 0.0;
	for (int32_t fi = 0; fi < m_factors->get_num_elements(); ++fi)
	{
		auto fac = m_factors->get_element<Factor>(fi);
		energy += fac->evaluate_energy(state);

	}
	return energy;
}

float64_t FactorGraph::evaluate_energy(std::shared_ptr<const FactorGraphObservation> obs) const
{
	return evaluate_energy(obs->get_data());
}

SGVector<float64_t> FactorGraph::evaluate_energies() const
{
	int num_assig = 1;
	SGVector<int32_t> cumprod_cards(m_cardinalities.size());
	for (int32_t n = 0; n < m_cardinalities.size(); ++n)
	{
		cumprod_cards[n] = num_assig;
		num_assig *= m_cardinalities[n];
	}

	SGVector<float64_t> etable(num_assig);
	for (int32_t ei = 0; ei < num_assig; ++ei)
	{
		SGVector<int32_t> assig(m_cardinalities.size());
		for (int32_t vi = 0; vi < m_cardinalities.size(); ++vi)
			assig[vi] = (ei / cumprod_cards[vi]) % m_cardinalities[vi];

		etable[ei] = evaluate_energy(assig);

		for (int32_t vi = 0; vi < m_cardinalities.size(); ++vi)
			SG_SPRINT("%d ", assig[vi]);

		SG_SPRINT("| %f\n", etable[ei]);
	}

	return etable;
}

void FactorGraph::connect_components()
{
	if (m_dset->get_connected())
		return;

	// need to be reset once factor graph is updated
	m_dset->make_sets();
	bool flag = false;

	for (int32_t fi = 0; fi < m_factors->get_num_elements(); ++fi)
	{
		auto fac = m_factors->get_element<Factor>(fi);
		SGVector<int32_t> vars = fac->get_variables();

		int32_t r0 = m_dset->find_set(vars[0]);
		for (int32_t vi = 1; vi < vars.size(); vi++)
		{
			// for two nodes in a factor, should be an edge between them
			// but this time link() isn't performed, if they are linked already
			// means there is another path connected them, so cycle detected
			int32_t ri = m_dset->find_set(vars[vi]);

			if (r0 == ri)
			{
				flag = true;
				continue;
			}

			r0 = m_dset->link_set(r0, ri);
		}


	}
	m_has_cycle = flag;
	m_dset->set_connected(true);
}

bool FactorGraph::is_acyclic_graph() const
{
	return !m_has_cycle;
}

bool FactorGraph::is_connected_graph() const
{
	return (m_dset->get_num_sets() == 1);
}

bool FactorGraph::is_tree_graph() const
{
	return (m_has_cycle == false && m_dset->get_num_sets() == 1);
}

void FactorGraph::loss_augmentation(std::shared_ptr<FactorGraphObservation> gt)
{
	loss_augmentation(gt->get_data(), gt->get_loss_weights());
}

void FactorGraph::loss_augmentation(SGVector<int32_t> states_gt, SGVector<float64_t> loss)
{
	if (loss.size() == 0)
	{
		loss.resize_vector(states_gt.size());
		SGVector<float64_t>::fill_vector(loss.vector, loss.vlen, 1.0 / states_gt.size());
	}

	int32_t num_vars = states_gt.size();
	ASSERT(num_vars == loss.size());

	SGVector<int32_t> var_flags(num_vars);
	var_flags.zero();

	// augment loss to incorrect states in the first factor containing the variable
	// since += L_i for each variable if it takes wrong state ever
	// TODO: augment unary factors
	for (int32_t fi = 0; fi < m_factors->get_num_elements(); ++fi)
	{
		auto fac = m_factors->get_element<Factor>(fi);
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
				auto ftype = fac->get_factor_type();
				int32_t vstate = ftype->state_from_index(ei, vi);


				if (states_gt[vv] == vstate)
					continue;

				// -delta(y_n, y_i_n)
				fac->set_energy(ei, energies[ei] - loss[vv]);
			}

			var_flags[vv] = 1;
		}


	}

	// make sure all variables have been checked
	int32_t min_var = Math::min(var_flags.vector, var_flags.vlen);
	ASSERT(min_var == 1);
}

