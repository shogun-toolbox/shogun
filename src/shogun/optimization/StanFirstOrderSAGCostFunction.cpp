/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Elfarouk
 */

#include <shogun/optimization/StanFirstOrderSAGCostFunction.h>
#include <shogun/base/range.h>
#include <shogun/mathematics/Math.h>
using namespace shogun;
using stan::math::var;
using std::function;
using Eigen::Matrix;
using Eigen::Dynamic;

void StanFirstOrderSAGCostFunction::set_training_data(
    SGMatrix<float64_t> X_new, SGMatrix<float64_t> y_new)
{
	REQUIRE(X_new.size() > 0, "Empty X provided");
	REQUIRE(y_new.size() > 0, "Empty y provided");
	this->m_X = X_new;
	this->m_y = y_new;
}

StanFirstOrderSAGCostFunction::~StanFirstOrderSAGCostFunction()
{
}

void StanFirstOrderSAGCostFunction::begin_sample()
{
	m_index_of_sample = -1;
}

bool StanFirstOrderSAGCostFunction::next_sample()
{
	++m_index_of_sample;
	return m_index_of_sample < get_sample_size();
}

void StanFirstOrderSAGCostFunction::update_stan_vectors_to_reference_values()
{
	auto num_of_variables = m_trainable_parameters.rows();
	for (auto i : range(num_of_variables))
	{
		m_trainable_parameters(i, 0) = m_ref_trainable_parameters[i];
	}
}
SGVector<float64_t> StanFirstOrderSAGCostFunction::get_gradient()
{
	auto num_of_variables = m_trainable_parameters.rows();
	REQUIRE(
	    num_of_variables > 0,
	    "Number of sample must be greater than 0, you provided no samples");

	update_stan_vectors_to_reference_values();
	var f_i = m_cost_for_ith_point(m_index_of_sample, 0)(
	    m_trainable_parameters, m_index_of_sample);

	stan::math::set_zero_all_adjoints();
	f_i.grad();

	SGVector<float64_t>::EigenVectorXt gradients =
	    m_trainable_parameters.unaryExpr(
	        [](stan::math::var x) -> float64_t { return x.adj(); });
	return SGVector<float64_t>(gradients).clone();
}

float64_t StanFirstOrderSAGCostFunction::get_cost()
{
	auto n = get_sample_size();
	StanVector cost_argument(n);

	update_stan_vectors_to_reference_values();
	for (auto i : range(n))
	{
		cost_argument(i, 0) =
		    m_cost_for_ith_point(i, 0)(m_trainable_parameters, i);
	}
	var cost = m_total_cost(cost_argument);
	return cost.val();
}

index_t StanFirstOrderSAGCostFunction::get_sample_size()
{
	return m_X.num_cols;
}

SGVector<float64_t> StanFirstOrderSAGCostFunction::get_average_gradient()
{
	auto params_num = m_trainable_parameters.rows();
	SGVector<float64_t> average_gradients(params_num);

	auto old_index_sample = m_index_of_sample;
	auto n = get_sample_size();
	REQUIRE(
	    n > 0,
	    "Number of sample must be greater than 0, you provided no samples");

	for (index_t i = 0; i < n; ++i)
	{
		m_index_of_sample = i;
		average_gradients += get_gradient();
	}
  average_gradients.scale(1.0/n);
  //lingalg::scale(average_gradients, 1.0/n);
	m_index_of_sample = old_index_sample;
	return average_gradients;
}

SGVector<float64_t> StanFirstOrderSAGCostFunction::obtain_variable_reference()
{
	return m_ref_trainable_parameters;
}
