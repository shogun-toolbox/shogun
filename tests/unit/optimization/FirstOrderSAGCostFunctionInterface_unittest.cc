/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Elfarouk
 */

#include <gtest/gtest.h>
#include <shogun/optimization/FirstOrderSAGCostFunctionInterface.h>
#include "FirstOrderSAGCostFunctionInterface_unittest.h"
#include <Eigen/Dense>

using namespace shogun;
using namespace Eigen;

/** This is a temporary fix. The variables are now stan::math::variables
*   So we cannot return a reference to the minimizer in the format
*   SGVector< stan::math::Var > until we update the minimizer's interface
*/
SGVector<float64_t> LeastSquareTestCostFunction::obtain_variable_reference()
{
	int32_t params_num = m_X->cols();
	SGVector<float64_t> ret(params_num);
	for (auto i = 0; i < params_num; ++i)
	{
		ret[i] = (*m_trainable_parameters)[i].val();
	}
	return ret;
}

TEST(LeastSquareTestCostFunction, ONALINE)
{
	int n = 3;
	auto X = Matrix<float64_t, Dynamic, Dynamic>();
	X.resize(2, 3);

	X(0, 0) = 1;
	X(0, 1) = 1;
	X(0, 2) = 1;
	X(1, 0) = 0;
	X(1, 1) = 1;
	X(1, 2) = 2;

	auto y = Matrix<float64_t, 1, Dynamic>();
	y.resize(1, 3);
	y(0, 0) = 0;
	y(0, 1) = 1;
	y(0, 2) = 2;

	auto w = Matrix<stan::math::var, Dynamic, 1>();
	stan::math::var m(1), c(0);
	w.resize(2, 1);
	w(0, 0) = c;
	w(1, 0) = m;

	auto f_i = [X, y, w](int32_t idx) {
		auto X_i = (X.col(idx));
		auto y_i = (y.col(idx));

		stan::math::var wx_y =
		    w(0, 0) * X_i(0, 0) + w(1, 0) * X_i(1, 0) - y_i(0, 0);
		stan::math::var res = wx_y * wx_y;
		return res;
	};
	std::vector<std::function<stan::math::var(int32_t)>> cost_for_ith_point(
	    n, f_i);

	auto cost = [](std::vector<stan::math::var>* v) {
		stan::math::var total_cost = 0;
		int32_t params_num = v->size();
		for (auto i = 0; i < params_num; ++i)
		{
			total_cost += (*v)[i];
		}
		return total_cost;
	};

	std::function<stan::math::var(std::vector<stan::math::var>*)> total_cost;
	total_cost = cost;

	LeastSquareTestCostFunction lstcf;
	lstcf.set_training_data(&X, &y);
	lstcf.set_ith_cost_function(&cost_for_ith_point);
	lstcf.set_cost_function(&total_cost);
	lstcf.set_trainable_parameters(&w);

	EXPECT_NEAR(lstcf.get_cost(), 0.0, 1e-5);

	lstcf.begin_sample();
	auto grad = lstcf.get_gradient();
	EXPECT_NEAR(grad[0], 0.0, 1e-5);
	EXPECT_NEAR(grad[1], 0.0, 1e-5);

	grad = lstcf.get_average_gradient();
	EXPECT_NEAR(grad[0], 0.0, 1e-5);
	EXPECT_NEAR(grad[1], 0.0, 1e-5);
}

TEST(LeastSquareTestCostFunction, NOTONALINE)
{
	int n = 3;
	auto X = Matrix<float64_t, Dynamic, Dynamic>();
	X.resize(2, 3);

	X(0, 0) = 1;
	X(0, 1) = 1;
	X(0, 2) = 1;
	X(1, 0) = 0;
	X(1, 1) = 1;
	X(1, 2) = 2;

	auto y = Matrix<float64_t, 1, Dynamic>();
	y.resize(1, 3);
	y(0, 0) = 0;
	y(0, 1) = 1;
	y(0, 2) = 4; // quadratic

	auto w = Matrix<stan::math::var, Dynamic, 1>();
	stan::math::var m(1), c(0); // y=x
	w.resize(2, 1);
	w(0, 0) = c;
	w(1, 0) = m;

	auto f_i = [X, y, w](int32_t idx) {
		auto X_i = (X.col(idx));
		auto y_i = (y.col(idx));

		stan::math::var wx_y =
		    w(0, 0) * X_i(0, 0) + w(1, 0) * X_i(1, 0) - y_i(0, 0);
		stan::math::var res = wx_y * wx_y;
		res /= 2;
		return res;
	};
	std::vector<std::function<stan::math::var(int32_t)>> cost_for_ith_point(
	    n, f_i);

	auto cost = [](std::vector<stan::math::var>* v) {
		stan::math::var total_cost = 0;
		int32_t params_num = v->size();
		for (auto i = 0; i < params_num; ++i)
		{
			total_cost += (*v)[i];
		}
		return total_cost;
	};

	std::function<stan::math::var(std::vector<stan::math::var>*)> total_cost;
	total_cost = cost;

	LeastSquareTestCostFunction lstcf;
	lstcf.set_training_data(&X, &y);
	lstcf.set_ith_cost_function(&cost_for_ith_point);
	lstcf.set_cost_function(&total_cost);
	lstcf.set_trainable_parameters(&w);

	EXPECT_NEAR(lstcf.get_cost(), 2.0, 1e-5);

	lstcf.begin_sample();
	lstcf.next_sample();
	lstcf.next_sample(); // third point
	auto grad = lstcf.get_gradient();
	EXPECT_NEAR(grad[0], -2.0, 1e-5);
	EXPECT_NEAR(grad[1], -4.0, 1e-5);

	grad = lstcf.get_average_gradient();
	EXPECT_NEAR(grad[0], -0.6666666667, 1e-5);
	EXPECT_NEAR(grad[1], -1.3333333333, 1e-5);
}
