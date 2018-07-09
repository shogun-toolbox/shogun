/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Elfarouk
 */

#include <shogun/optimization/StanFirstOrderSAGCostFunction.h>
#include "StanFirstOrderSAGCostFunction_unittest.h"
#include <Eigen/Dense>
#include <gtest/gtest.h>

using namespace shogun;
using Eigen::Matrix;
using Eigen::Dynamic;
using stan::math::var;
using std::function;

SGMatrix<float64_t> generateXSmall()
{
	SGMatrix<float64_t> X(2, 3);

	X(0, 0) = 1;
	X(0, 1) = 1;
	X(0, 2) = 1;
	X(1, 0) = 0;
	X(1, 1) = 1;
	X(1, 2) = 2;

	return X;
}

StanVector generateParametersSmall()
{
	StanVector w(2, 1);
	var m(1), c(0);
	w(0, 0) = c;
	w(1, 0) = m;

	return w;
}

function<var(const StanVector&, float64_t)>
cost_for_ith_datapoint(SGMatrix<float64_t>& X, SGMatrix<float64_t>& y)
{
	auto f_i = [X, y](const StanVector& w, int32_t idx) {
		var wx_y = w(0, 0) * X(0, idx) + w(1, 0) * X(1, idx) - y(0, idx);
		var res = wx_y * wx_y;
		res *= 0.5;
		return res;
	};
	return f_i;
}

function<var(const StanVector&)> get_total_cost()
{
	auto cost = [](const StanVector& v) {
		var total_cost = v.sum();
		return total_cost;
	};
	return cost;
}

TEST(LeastSquareTestCostFunction, points_on_a_line)
{
	int n = 3;

	auto X = generateXSmall();

	SGMatrix<float64_t> y(1, 3);
	y(0, 0) = 0;
	y(0, 1) = 1;
	y(0, 2) = 2;

	auto w = generateParametersSmall();

	auto f_i = cost_for_ith_datapoint(X, y);

	Matrix<function<var(const StanVector&, float64_t)>, Dynamic, 1>
	    cost_for_ith_point =
	        Matrix<function<var(const StanVector&, float64_t)>, Dynamic, 1>::Constant(
	            n, 1, f_i);

	auto total_cost = get_total_cost();

	LeastSquareTestCostFunction lstcf(
	    X, y, w, cost_for_ith_point, total_cost);

	EXPECT_NEAR(lstcf.get_cost(), 0.0, 1e-5);

	lstcf.begin_sample();
	lstcf.next_sample();
	auto grad = lstcf.get_gradient();
	EXPECT_NEAR(grad[0], 0.0, 1e-5);
	EXPECT_NEAR(grad[1], 0.0, 1e-5);

	grad = lstcf.get_average_gradient();
	EXPECT_NEAR(grad[0], 0.0, 1e-5);
	EXPECT_NEAR(grad[1], 0.0, 1e-5);
}

TEST(LeastSquareTestCostFunction, points_on_y_equals_x_squared)
{
	int n = 3;

	auto X = generateXSmall();

	SGMatrix<float64_t> y(1, 3);
	y(0, 0) = 0;
	y(0, 1) = 1;
	y(0, 2) = 4; // y=x^2

	auto w = generateParametersSmall();

	auto f_i = cost_for_ith_datapoint(X, y);

	Matrix<function<var(const StanVector&, float64_t)>, Dynamic, 1>
	    cost_for_ith_point =
	        Matrix<function<var(const StanVector&, float64_t)>, Dynamic, 1>::Constant(
	            n, 1, f_i);

	auto total_cost = get_total_cost();

	LeastSquareTestCostFunction lstcf(
	    X, y, w, cost_for_ith_point, total_cost);

	EXPECT_NEAR(lstcf.get_cost(), 2.0, 1e-5);

	lstcf.begin_sample();
	lstcf.next_sample();
	lstcf.next_sample();
	lstcf.next_sample(); // third point
	auto grad = lstcf.get_gradient();
	EXPECT_NEAR(grad[0], -2.0, 1e-5);
	EXPECT_NEAR(grad[1], -4.0, 1e-5);

	grad = lstcf.get_average_gradient();
	EXPECT_NEAR(grad[0], -0.6666666667, 1e-5);
	EXPECT_NEAR(grad[1], -1.3333333333, 1e-5);
}
