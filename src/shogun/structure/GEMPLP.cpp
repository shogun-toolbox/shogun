/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Jiaolong Xu, Sanuj Sharma, Bjoern Esser, Yori Zwols
 */

#include <shogun/structure/GEMPLP.h>
#include <shogun/io/SGIO.h>
#include <algorithm>

using namespace shogun;
using namespace std;

GEMPLP::GEMPLP()
	: MAPInferImpl()
{
	m_fg = NULL;
	m_factors = NULL;
}

GEMPLP::GEMPLP(std::shared_ptr<FactorGraph> fg, Parameter param)
	: MAPInferImpl(fg),
	  m_param(param)
{
	ASSERT(m_fg != NULL);

	init();
}

GEMPLP::~GEMPLP()
{

}

void GEMPLP::init()
{
	SGVector<int32_t> fg_var_sizes = m_fg->get_cardinalities();
	m_factors = m_fg->get_factors();

	int32_t num_factors = m_factors->get_num_elements();
	m_region_intersections.resize(num_factors);

	// get all the intersections
	for (int32_t i = 0; i < num_factors; i++)
	{
		auto factor_i = m_factors->get_element<Factor>(i);
		SGVector<int32_t> region_i = factor_i->get_variables();


		for (int32_t j = i; j < num_factors; j++)
		{
			auto factor_j = m_factors->get_element<Factor>(j);
			SGVector<int32_t> region_j = factor_j->get_variables();


			const int32_t k = find_intersection_index(region_i, region_j);
			if (k < 0) continue;

			m_region_intersections[i].insert(k);

			if (j != i)
				m_region_intersections[j].insert(k);
		}
	}

	m_region_inds_intersections.resize(num_factors);
	m_msgs_from_region.resize(num_factors);
	m_theta_region.resize(num_factors);

	for (int32_t c = 0; c < num_factors; c++)
	{
		auto factor_c = m_factors->get_element<Factor>(c);
		SGVector<int32_t> vars_c = factor_c->get_variables();


		m_region_inds_intersections[c].resize(m_region_intersections[c].size());
		m_msgs_from_region[c].resize(m_region_intersections[c].size());

		int32_t s = 0;

		for (std::set<int>::iterator t = m_region_intersections[c].begin();
				t != m_region_intersections[c].end(); t++)
		{
			SGVector<int32_t> curr_intersection = m_all_intersections[*t];
			SGVector<int32_t> inds_s(curr_intersection.size());
			SGVector<int32_t> dims_array(curr_intersection.size());

			for (int32_t i = 0; i < inds_s.size(); i++)
			{
				inds_s[i] = vars_c.find(curr_intersection[i])[0];
				REQUIRE(inds_s[i] >= 0,
						"Intersection contains variable %d which is not in the region %d", curr_intersection[i], c);

				dims_array[i] = fg_var_sizes[curr_intersection[i]];
			}

			// initialize indices of intersections inside the region
			m_region_inds_intersections[c][s] = inds_s;

			// initialize messages from region and set it 0
			SGNDArray<float64_t> message(dims_array);
			message.set_const(0);
			m_msgs_from_region[c][s] = message.clone();
			s++;
		}

		// initialize potential on region
		m_theta_region[c] = convert_energy_to_potential(factor_c);
	}

	// initialize messages in intersections and set it 0
	m_msgs_into_intersections.resize(m_all_intersections.size());

	for (uint32_t i = 0; i < m_all_intersections.size(); i++)
	{
		SGVector<int32_t> vars_intersection = m_all_intersections[i];
		SGVector<int32_t> dims_array(vars_intersection.size());

		for (int32_t j = 0; j < dims_array.size(); j++)
			dims_array[j] = fg_var_sizes[vars_intersection[j]];

		SGNDArray<float64_t> curr_array(dims_array);
		curr_array.set_const(0);
		m_msgs_into_intersections[i] = curr_array.clone();
	}
}

SGNDArray<float64_t> GEMPLP::convert_energy_to_potential(std::shared_ptr<Factor> factor)
{
	SGVector<float64_t> energies = factor->get_energies();
	SGVector<int32_t> cards = factor->get_cardinalities();

	SGNDArray<float64_t> message(cards);

	if (cards.size() == 1)
	{
		for (int32_t i = 0; i < energies.size(); i++)
			message.array[i] = - energies[i];
	}
	else if (cards.size() == 2)
	{
		for (int32_t y = 0; y < cards[1]; y++)
			for (int32_t x = 0; x < cards[0]; x++)
				message.array[x*cards[1]+y] = - energies[y*cards[0]+x];
	}
	else
		SG_ERROR("Index issue has not been solved for higher order (>=3) factors.");

	return message.clone();
}

int32_t GEMPLP::find_intersection_index(SGVector<int32_t> region_A, SGVector<int32_t> region_B)
{
	vector<int32_t> tmp;

	for (int32_t i = 0; i < region_A.size(); i++)
	{
		for (int32_t j = 0; j < region_B.size(); j++)
		{
			if (region_A[i] == region_B[j])
				tmp.push_back(region_A[i]);
		}
	}

	// return -1 if intersetion is empty
	if (tmp.size() == 0)	return -1;


	SGVector<int32_t> sAB(tmp.size());
	for (uint32_t i = 0; i < tmp.size(); i++)
		sAB[i] = tmp[i];

	// find (or add) intersection set
	int32_t k;
	for (k = 0; k < (int32_t)m_all_intersections.size(); k++)
		if (m_all_intersections[k].equals(sAB))
			break;

	if (k == (int32_t)m_all_intersections.size())
		m_all_intersections.push_back(sAB);

	return k;
}

float64_t GEMPLP::inference(SGVector<int32_t> assignment)
{
	REQUIRE(assignment.size() == m_fg->get_cardinalities().size(),
	        "%s::inference(): the output assignment should be prepared as"
	        "the same size as variables!\n", get_name());

	// iterate over message loop
	SG_SDEBUG("Running MPLP for %d iterations\n",  m_param.m_max_iter);

	float64_t last_obj = Math::INFTY;

	// block coordinate desent, outer loop
	for (int32_t it = 0; it < m_param.m_max_iter; ++it)
	{
		// update message, iterate over all regions
		for (int32_t c = 0; c < m_factors->get_num_elements(); c++)
		{
			auto factor_c =m_factors->get_element<Factor>(c);
			SGVector<int32_t> vars = factor_c->get_variables();


			if (vars.size() == 1 && it > 0)
				continue;

			update_messages(c);
		}

		// calculate the objective value
		float64_t obj = 0;
		int32_t max_at;

		for (uint32_t s = 0; s < m_msgs_into_intersections.size(); s++)
		{
			obj += m_msgs_into_intersections[s].max_element(max_at);

			if (m_all_intersections[s].size() == 1)
				assignment[m_all_intersections[s][0]] = max_at;
		}

		// get the value of the decoded solution
		float64_t int_val = 0;

		// iterates over factors
		for (int32_t c = 0; c < m_factors->get_num_elements(); c++)
		{
			auto factor = m_factors->get_element<Factor>(c);
			SGVector<int32_t> vars = factor->get_variables();
			SGVector<int32_t> var_assignment(vars.size());

			for (int32_t i = 0; i < vars.size(); i++)
				var_assignment[i] = assignment[vars[i]];

			// add value from current factor
			int_val += m_theta_region[c].get_value(var_assignment);


		}

		float64_t obj_del = last_obj - obj;
		float64_t int_gap = obj - int_val;

		SG_SDEBUG("Iter= %d Objective=%f ObjBest=%f ObjDel=%f Gap=%f \n", (it + 1), obj, int_val, obj_del, int_gap);

		if (obj_del < m_param.m_obj_del_thr && it > 16)
			break;

		if (int_gap < m_param.m_int_gap_thr)
			break;

		last_obj = obj;
	}

	float64_t energy = m_fg->evaluate_energy(assignment);
	SG_DEBUG("fg.evaluate_energy(assignment) = %f\n", energy);

	return energy;
}

void GEMPLP::update_messages(int32_t id_region)
{
	REQUIRE(m_factors != NULL, "Factors are not set!\n");

	REQUIRE(m_factors->get_num_elements() > id_region,
			"Region id (%d) exceeds the factor elements' length (%d)!\n",
			id_region, m_factors->get_num_elements());

	auto factor = m_factors->get_element<Factor>(id_region);
	SGVector<int32_t> vars = factor->get_variables();
	SGVector<int32_t> cards = factor->get_cardinalities();
	SGNDArray<float64_t> lam_sum(cards);

	if (m_theta_region[id_region].len_array == 0)
		lam_sum.set_const(0);
	else
		lam_sum = m_theta_region[id_region].clone();

	int32_t num_intersections = m_region_intersections[id_region].size();
	vector<SGNDArray<float64_t> > lam_minus; // substract message: \lambda_s^{-c}(x_s)
	// \sum_{\hat{s}} \lambda_{\hat{s}}^{-c}(x_{\hat{s}}) + \theta_c(x_c)
	int32_t s = 0;

	for (std::set<int32_t>::iterator t = m_region_intersections[id_region].begin();
				t != m_region_intersections[id_region].end(); t++)
	{
		int32_t id_intersection = *t;
		SGNDArray<float64_t> tmp = m_msgs_into_intersections[id_intersection].clone();
		tmp -= m_msgs_from_region[id_region][s];

		lam_minus.push_back(tmp);

		if (vars.size() == (int32_t)m_region_inds_intersections[id_region][s].size())
			lam_sum += tmp;
		else
		{
			SGNDArray<float64_t> tmp_expand(lam_sum.get_dimensions());
			tmp.expand(tmp_expand, m_region_inds_intersections[id_region][s]);
			lam_sum += tmp_expand;
		}

		// take out the old incoming message: \lambda_{c \to s}(x_s)
		m_msgs_into_intersections[id_intersection] -= m_msgs_from_region[id_region][s];
		s++;
	}

	s = 0;

	for (std::set<int32_t>::iterator t = m_region_intersections[id_region].begin();
				t != m_region_intersections[id_region].end(); t++)
	{
		// maximazation: \max_{x_c} \sum_{\hat{s}} \lambda_{\hat{s}}^{-c}(x_{\hat{s}}) + \theta_c(x_c)
		SGNDArray<float64_t> lam_max(lam_minus[s].get_dimensions());
		max_in_subdimension(lam_sum, m_region_inds_intersections[id_region][s], lam_max);
		int32_t id_intersection = *t;
		// weighted sum
		lam_max *= 1.0/num_intersections;
		m_msgs_from_region[id_region][s] = lam_max.clone();
		m_msgs_from_region[id_region][s] -= lam_minus[s];
		// put in new message
		m_msgs_into_intersections[id_intersection] += m_msgs_from_region[id_region][s];
		s++;
	}


}

void GEMPLP::max_in_subdimension(SGNDArray<float64_t> tar_arr, SGVector<int32_t> &subset_inds, SGNDArray<float64_t> &max_res) const
{
	// If the subset equals the target array then maximizing would
	// give us the target array (assuming there is no reordering)
	if (subset_inds.size() == tar_arr.num_dims)
	{
		max_res = tar_arr.clone();
		return;
	}
	else
		max_res.set_const(-Math::INFTY);

	// Go over all values of the target array. For each check if its
	// value on the subset is larger than the current max
	SGVector<int32_t> inds_for_tar(tar_arr.num_dims);
	inds_for_tar.zero();

	for (int32_t vi = 0; vi < tar_arr.len_array; vi++)
	{
		int32_t y = 0;

		if (subset_inds.size() == 1)
			y = inds_for_tar[subset_inds[0]];
		else if (subset_inds.size() == 2)
		{
			int32_t ind1 = subset_inds[0];
			int32_t ind2 = subset_inds[1];
			y = inds_for_tar[ind1] * max_res.dims[1] + inds_for_tar[ind2];
		}
		max_res[y] = max(max_res[y], tar_arr.array[vi]);
		tar_arr.next_index(inds_for_tar);
	}
}
