/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Shell Hu, Heiko Strathmann, Jiaolong Xu, Bjoern Esser,
 *          Sergey Lisitsyn
 */

#include <shogun/structure/Factor.h>
#include <shogun/base/Parameter.h>

using namespace shogun;

Factor::Factor() : SGObject()
{
	init();
}

Factor::Factor(std::shared_ptr<TableFactorType> ftype,
	SGVector<int32_t> var_index,
	SGVector<float64_t> data) : SGObject()
{
	init();
	m_factor_type = ftype;
	m_var_index = var_index;
	m_data = data;
	m_is_data_dep = true;

	ASSERT(m_factor_type != NULL);
	ASSERT(m_factor_type->get_cardinalities().size() == m_var_index.size());

	if (m_data.size() == 0)
		m_is_data_dep = false;

	if (ftype->is_table() && m_is_data_dep)
		m_energies.resize_vector(ftype->get_num_assignments());



}

Factor::Factor(std::shared_ptr<TableFactorType> ftype,
	SGVector<int32_t> var_index,
	SGSparseVector<float64_t> data_sparse) : SGObject()
{
	init();
	m_factor_type = ftype;
	m_var_index = var_index;
	m_data_sparse = data_sparse;
	m_is_data_dep = true;

	ASSERT(m_factor_type != NULL);
	ASSERT(m_factor_type->get_cardinalities().size() == m_var_index.size());

	if (m_data_sparse.num_feat_entries == 0)
		m_is_data_dep = false;

	if (ftype->is_table() && m_is_data_dep)
		m_energies.resize_vector(ftype->get_num_assignments());



}

Factor::Factor(std::shared_ptr<TableFactorType> ftype,
	SGVector<int32_t> var_index,
	std::shared_ptr<FactorDataSource> data_source) : SGObject()
{
	init();
	m_factor_type = ftype;
	m_var_index = var_index;
	m_data_source = data_source;
	m_is_data_dep = true;

	ASSERT(m_factor_type != NULL);
	ASSERT(m_factor_type->get_cardinalities().size() == m_var_index.size());
	ASSERT(m_data_source != NULL);

	if (ftype->is_table())
		m_energies.resize_vector(ftype->get_num_assignments());



}

Factor::~Factor()
{


}

std::shared_ptr<TableFactorType> Factor::get_factor_type() const
{

	return m_factor_type;
}

void Factor::set_factor_type(std::shared_ptr<TableFactorType> ftype)
{
	m_factor_type = ftype;

}

const SGVector<int32_t> Factor::get_variables() const
{
	return m_var_index;
}

const int32_t Factor::get_num_vars() const
{
	return m_var_index.size();
}

void Factor::set_variables(SGVector<int32_t> vars)
{
	m_var_index = vars.clone();
}

const SGVector<int32_t> Factor::get_cardinalities() const
{
	return m_factor_type->get_cardinalities();
}

SGVector<float64_t> Factor::get_data() const
{
	if (m_data_source != NULL)
		return m_data_source->get_data();

	return m_data;
}

SGSparseVector<float64_t> Factor::get_data_sparse() const
{
	if (m_data_source != NULL)
		return m_data_source->get_data_sparse();

	return m_data_sparse;
}

void Factor::set_data(SGVector<float64_t> data_dense)
{
	m_data = data_dense.clone();
	m_is_data_dep = true;
}

void Factor::set_data_sparse(SGSparseVectorEntry<float64_t>* data_sparse,
	int32_t dlen)
{
	m_data_sparse = SGSparseVector<float64_t>(data_sparse, dlen);
	m_is_data_dep = true;
}

bool Factor::is_data_dependent() const
{
	return m_is_data_dep;
}

bool Factor::is_data_sparse() const
{
	if (m_data_source != NULL)
		return m_data_source->is_sparse();

	return (m_data.size() == 0);
}

SGVector<float64_t> Factor::get_energies() const
{
	if (is_data_dependent() == false && m_energies.size() == 0)
	{
		const SGVector<float64_t> ft_energies = m_factor_type->get_w();
		ASSERT(ft_energies.size() == m_factor_type->get_num_assignments());
		return ft_energies;
	}
	return m_energies;
}

float64_t Factor::get_energy(int32_t index) const
{
	return get_energies()[index]; // note for data indep, we get m_w not m_energies
}

void Factor::set_energies(SGVector<float64_t> ft_energies)
{
	REQUIRE(m_factor_type->get_num_assignments() == ft_energies.size(),
		"%s::set_energies(): ft_energies is not a valid energy table!\n", get_name());

	m_energies = ft_energies;
}

void Factor::set_energy(int32_t ei, float64_t value)
{
	REQUIRE(ei >= 0 && ei < m_factor_type->get_num_assignments(),
		"%s::set_energy(): ei is out of index!\n", get_name());

	REQUIRE(is_data_dependent(), "%s::set_energy(): \
		energy table is fixed in data dependent factor!\n", get_name());

	m_energies[ei] = value;
}

float64_t Factor::evaluate_energy(const SGVector<int32_t> state) const
{
	int32_t index = m_factor_type->index_from_universe_assignment(state, m_var_index);
	return get_energy(index);
}

void Factor::compute_energies()
{
	if (is_data_dependent() == false)
		return;

	// For some factor types the size of the energy table is determined only
	// after an initialization step from training data.
	if (m_energies.size() == 0)
		m_energies.resize_vector(m_factor_type->get_num_assignments());

	const SGVector<float64_t> H = get_data();
	const SGSparseVector<float64_t> H_sparse = get_data_sparse();

	if (H_sparse.num_feat_entries == 0)
		m_factor_type->compute_energies(H, m_energies);
	else
		m_factor_type->compute_energies(H_sparse, m_energies);
}

void Factor::compute_gradients(
	const SGVector<float64_t> marginals,
	SGVector<float64_t>& parameter_gradient,
	float64_t mult) const
{
	const SGVector<float64_t> H = get_data();
	const SGSparseVector<float64_t> H_sparse = get_data_sparse();

	if (H_sparse.num_feat_entries == 0)
		m_factor_type->compute_gradients(H, marginals, parameter_gradient, mult);
	else
		m_factor_type->compute_gradients(H_sparse, marginals, parameter_gradient, mult);
}

void Factor::init()
{
	SG_ADD((std::shared_ptr<SGObject>*)&m_factor_type, "type_name", "Factor type name");
	SG_ADD(&m_var_index, "var_index", "Factor variable index");
	SG_ADD(&m_energies, "energies", "Factor energies");
	SG_ADD((std::shared_ptr<SGObject>*)&m_data_source, "data_source", "Factor data source");
	SG_ADD(&m_data, "data", "Factor data");
	SG_ADD(&m_data_sparse, "data_sparse", "Sparse factor data");
	SG_ADD(&m_is_data_dep, "is_data_dep", "Factor is data dependent or not");

	m_factor_type=NULL;
	m_data_source=NULL;
	m_is_data_dep = false;
}

FactorDataSource::FactorDataSource() : SGObject()
{
	init();
}

FactorDataSource::FactorDataSource(SGVector<float64_t> dense)
	: SGObject()
{
	init();
	m_dense = dense;
}

FactorDataSource::FactorDataSource(SGSparseVector<float64_t> sparse)
	: SGObject()
{
	init();
	m_sparse = sparse;
}

FactorDataSource::~FactorDataSource()
{
}

bool FactorDataSource::is_sparse() const
{
	return (m_dense.size() == 0);
}

SGVector<float64_t> FactorDataSource::get_data() const
{
	return m_dense;
}

SGSparseVector<float64_t> FactorDataSource::get_data_sparse() const
{
	return m_sparse;
}

void FactorDataSource::set_data(SGVector<float64_t> dense)
{
	m_dense = dense.clone();
}

void FactorDataSource::set_data_sparse(SGSparseVectorEntry<float64_t>* sparse,
	int32_t dlen)
{
	m_sparse = SGSparseVector<float64_t>(sparse, dlen);
}

void FactorDataSource::init()
{
	SG_ADD(&m_dense, "dense", "Shared data");
	SG_ADD(&m_sparse, "sparse", "Shared sparse data");
}

