/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Elfarouk
 */

#include <shogun/optimization/FirstOrderSAGCostFunctionInterface.h>
#include <shogun/mathematics/Math.h>
using namespace shogun;
using stan::math::var;
using std::function;
using Eigen::Matrix;

FirstOrderSAGCostFunctionInterface::FirstOrderSAGCostFunctionInterface(
    SGMatrix<float64_t>* X, SGMatrix<float64_t>* y,
    Matrix<var, Dynamic, 1>* trainable_parameters,
    Matrix<function<var(int32_t)>, Dynamic, 1>* cost_for_ith_point,
    function<var(Matrix<var, Dynamic, 1>*)>* total_cost)
{
	m_X = X;
	m_y = y;
	m_trainable_parameters = trainable_parameters;
	m_cost_for_ith_point = cost_for_ith_point;
	m_total_cost = total_cost;
}

void FirstOrderSAGCostFunctionInterface::set_training_data(
    SGMatrix<float64_t>* X_new, SGMatrix<float64_t>* y_new)
{
	REQUIRE(X_new != NULL, "No X Provided");
	REQUIRE(y_new != NULL, "No Y Provided");
	if (this->m_X != X_new)
	{
		this->m_X = X_new;
	}
	if (this->m_y != y_new)
	{
		this->m_y = y_new;
	}
}

void FirstOrderSAGCostFunctionInterface::set_trainable_parameters(
    Matrix<var, Dynamic, 1>* new_params)
{
	REQUIRE(new_params, "The trainable parameters must be provided");
	if (this->m_trainable_parameters != new_params)
	{
		this->m_trainable_parameters = new_params;
	}
}

void FirstOrderSAGCostFunctionInterface::set_ith_cost_function(
    Matrix<function<var(int32_t)>, Dynamic, 1>* new_cost_f)
{
	REQUIRE(new_cost_f, "The cost function must be a vector of stan variables");
	if (this->m_cost_for_ith_point != new_cost_f)
	{
		this->m_cost_for_ith_point = new_cost_f;
	}
}

void FirstOrderSAGCostFunctionInterface::set_cost_function(
    function<var(Matrix<var, Dynamic, 1>*)>* total_cost)
{
	REQUIRE(
	    total_cost,
	    "The total cost function must be a function returning a stan variable");
	if (this->m_total_cost != total_cost)
	{
		this->m_total_cost = total_cost;
	}
}

FirstOrderSAGCostFunctionInterface::~FirstOrderSAGCostFunctionInterface()
{
}

void FirstOrderSAGCostFunctionInterface::begin_sample()
{
	m_index_of_sample = 0;
}

bool FirstOrderSAGCostFunctionInterface::next_sample()
{
	auto num_of_samples = get_sample_size();
	if (m_index_of_sample >= num_of_samples)
		return false;
	++m_index_of_sample;
	return true;
}

SGVector<float64_t> FirstOrderSAGCostFunctionInterface::get_gradient()
{
	int32_t num_of_variables = m_trainable_parameters->rows();
	REQUIRE(
	    num_of_variables > 0,
	    "Number of training parameters must be greater than 0");

	SGVector<float64_t> gradients(num_of_variables);

	var f_i = (*m_cost_for_ith_point)(m_index_of_sample, 0)(m_index_of_sample);

	stan::math::set_zero_all_adjoints();
	f_i.grad();

	for (auto i = 0; i < num_of_variables; ++i)
		gradients[i] = (*m_trainable_parameters)(i, 0).adj();

	return gradients;
}

float64_t FirstOrderSAGCostFunctionInterface::get_cost()
{
	int32_t n = get_sample_size();
	Matrix<var, Dynamic, 1> cost_argument(n);
	for (auto i = 0; i < n; ++i)
	{
		cost_argument(i, 0) = (*m_cost_for_ith_point)(i, 0)(i);
	}
	var cost = (*m_total_cost)(&cost_argument);
	return cost.val();
}

index_t FirstOrderSAGCostFunctionInterface::get_sample_size()
{
	return m_X->num_cols;
}

SGVector<float64_t> FirstOrderSAGCostFunctionInterface::get_average_gradient()
{
	int32_t params_num = m_trainable_parameters->rows();
	SGVector<float64_t> average_gradients(params_num);

	int32_t old_index_sample = m_index_of_sample;
	int32_t n = get_sample_size();
	REQUIRE(
	    n > 0,
	    "Number of sample must be greater than 0, you provided no samples");

	for (index_t i = 0; i < n; ++i)
	{
		m_index_of_sample = i;
		auto grad = get_gradient();
		average_gradients += grad;
	}
	average_gradients.scale(1.0 / n);
	m_index_of_sample = old_index_sample;
	return average_gradients;
}
