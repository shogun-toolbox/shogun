/*
 * Copyright (c) The Shogun Machine Learning Toolbox
 * Written (w) 2014 Wu Lin
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice, this
 *    list of conditions and the following disclaimer.
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 *    this list of conditions and the following disclaimer in the documentation
 *    and/or other materials provided with the distribution.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
 * ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
 * WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR
 * ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
 * (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 * LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
 * ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 * SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 * The views and conclusions contained in the software and documentation are those
 * of the authors and should not be interpreted as representing official policies,
 * either expressed or implied, of the Shogun Development Team.
 *
 *
 */
#include <shogun/lib/config.h>

#include <shogun/mathematics/eigen3.h>
#include <shogun/optimization/lbfgs/lbfgs.h>
#include <gtest/gtest.h>
#include <shogun/mathematics/Math.h>
#include <shogun/lib/SGMatrix.h>
#include <shogun/lib/SGVector.h>

using namespace shogun;
using namespace Eigen;

const float64_t upper_bound=12;
const float64_t lower_bound=1;
const float64_t strict_scale=1e-5;

//Init the parameters used for L-BFGS
lbfgs_parameter_t init_lbfgs_parameters()
{
	lbfgs_parameter_t tmp;
	tmp.m = 100;
	tmp.max_linesearch = 1000;
	tmp.linesearch = LBFGS_LINESEARCH_DEFAULT;
	tmp.max_iterations = 1000;
	tmp.delta = 1e-15;
	tmp.past = 0;
	tmp.epsilon = 1e-15;
	tmp.min_step = 1e-20;
	tmp.max_step = 1e+20;
	tmp.ftol = 1e-4;
	tmp.wolfe = 0.9;
	tmp.gtol = 0.9;
	tmp.xtol = 1e-16;
	tmp.orthantwise_c = 0;
	tmp.orthantwise_start = 0;
	tmp.orthantwise_end = 1;
	return tmp;
}

float64_t evaluate(void *obj, const float64_t *variable, float64_t *gradient,
	const int dim, const float64_t step)
{
	float64_t * non_const_variable=const_cast<float64_t *>(variable);
	const Map<VectorXd> eigen_x(non_const_variable, dim);
	float64_t f=eigen_x.array().pow(2.0).sum();

	Map<VectorXd> eigen_g(gradient, dim);
	eigen_g=eigen_x.array()*2.0;

	return f;
}

float64_t adjust_step_bounded(void *obj, const float64_t *parameters,
	const float64_t *direction, const int dim, const float64_t step)
{
	float64_t min_step=step;
	for (index_t i=0; i<dim; i++)
	{
		float64_t attempt=parameters[i]+step*direction[i];
		float64_t adjust=0;

		if (direction[i]==0.0)
			continue;
		if (attempt<lower_bound)
		{
			adjust=(parameters[i]-lower_bound)/CMath::abs(direction[i]);
			if (adjust<min_step)
				min_step=adjust;
		}

		if (attempt>upper_bound)
		{
			adjust=(upper_bound-parameters[i])/CMath::abs(direction[i]);
			if (adjust<min_step)
				min_step=adjust;
		}
	}
	return min_step;
}

float64_t evaluate_bounded(void *obj, const float64_t *variable, float64_t *gradient,
	const int dim, const float64_t step)
{
	bool is_valid=true;

	for (index_t i=0; i<dim; i++)
	{
		if (variable[i]>upper_bound || variable[i]<lower_bound)
		{
			is_valid=false;
			break;
		}
	}
	if (!is_valid)
		return CMath::INFTY;

	float64_t * non_const_variable=const_cast<float64_t *>(variable);
	const Map<VectorXd> eigen_x(non_const_variable, dim);
	float64_t f=eigen_x.array().pow(2.0).sum();

	Map<VectorXd> eigen_g(gradient, dim);
	eigen_g=eigen_x.array()*2.0;

	return f;
}

float64_t evaluate_strict_bounded(void *obj, const float64_t *variable, float64_t *gradient,
	const int dim, const float64_t step)
{
	bool is_valid=true;

	for (index_t i=0; i<dim; i++)
	{
		if (variable[i]>=upper_bound || variable[i]<=lower_bound)
		{
			is_valid=false;
			break;
		}
	}
	if (!is_valid)
		return CMath::INFTY;

	float64_t * non_const_variable=const_cast<float64_t *>(variable);
	const Map<VectorXd> eigen_x(non_const_variable, dim);
	float64_t f=eigen_x.array().pow(2.0).sum();

	Map<VectorXd> eigen_g(gradient, dim);
	eigen_g=eigen_x.array()*2.0;

	return f;
}

float64_t adjust_step_strict_bounded(void *obj, const float64_t *parameters,
	const float64_t *direction, const int dim, const float64_t step)
{
	float64_t min_step=step;
	for (index_t i=0; i<dim; i++)
	{
		float64_t attempt=parameters[i]+step*direction[i];
		float64_t adjust=0;

		if (direction[i]==0.0)
			continue;
		if (attempt<lower_bound)
		{
			adjust=(parameters[i]-lower_bound)/CMath::abs(direction[i]);
			adjust*=(1-strict_scale);
			if (adjust<min_step)
				min_step=adjust;
		}

		if (attempt>upper_bound)
		{
			adjust=(upper_bound-parameters[i])/CMath::abs(direction[i]);
			adjust*=(1-strict_scale);
			if (adjust<min_step)
				min_step=adjust;
		}
	}
	return min_step;
}

TEST(lbfgs, original_lbfgs)
{
	index_t len=5;
	SGVector<float64_t> x(len);
	Map<VectorXd> eigen_x(x.vector, x.vlen);
	eigen_x.fill(10);
	lbfgs_parameter_t lbfgs_param=init_lbfgs_parameters();
	float64_t opt_value=CMath::INFTY;
	lbfgs(x.vlen, x.vector, &opt_value,
		evaluate, NULL, NULL, &lbfgs_param);

	float64_t rel_tolerance = 1e-15;
	float64_t abs_tolerance;

	abs_tolerance = CMath::get_abs_tolerance(0, rel_tolerance);
	EXPECT_NEAR(opt_value, 0, abs_tolerance);

	abs_tolerance = CMath::get_abs_tolerance(0, rel_tolerance);
	EXPECT_NEAR(x[0], 0, abs_tolerance);

	abs_tolerance = CMath::get_abs_tolerance(0, rel_tolerance);
	EXPECT_NEAR(x[1], 0, abs_tolerance);

	abs_tolerance = CMath::get_abs_tolerance(0, rel_tolerance);
	EXPECT_NEAR(x[2], 0, abs_tolerance);

	abs_tolerance = CMath::get_abs_tolerance(0, rel_tolerance);
	EXPECT_NEAR(x[3], 0, abs_tolerance);

	abs_tolerance = CMath::get_abs_tolerance(0, rel_tolerance);
	EXPECT_NEAR(x[4], 0, abs_tolerance);
}

TEST(lbfgs, lbfgs_with_adjust_step_test1)
{
	index_t len=5;
	SGVector<float64_t> x(len);
	Map<VectorXd> eigen_x(x.vector, x.vlen);
	eigen_x.fill(10);
	lbfgs_parameter_t lbfgs_param=init_lbfgs_parameters();
	float64_t opt_value=CMath::INFTY;
	lbfgs(x.vlen, x.vector, &opt_value,
		evaluate_bounded, NULL, NULL, &lbfgs_param, &adjust_step_bounded);

	float64_t rel_tolerance = 1e-15;
	float64_t abs_tolerance;

	abs_tolerance = CMath::get_abs_tolerance(len, rel_tolerance);
	EXPECT_NEAR(opt_value, len, abs_tolerance);

	abs_tolerance = CMath::get_abs_tolerance(lower_bound, rel_tolerance);
	EXPECT_NEAR(x[0], lower_bound, abs_tolerance);

	abs_tolerance = CMath::get_abs_tolerance(lower_bound, rel_tolerance);
	EXPECT_NEAR(x[1], lower_bound, abs_tolerance);

	abs_tolerance = CMath::get_abs_tolerance(lower_bound, rel_tolerance);
	EXPECT_NEAR(x[2], lower_bound, abs_tolerance);

	abs_tolerance = CMath::get_abs_tolerance(lower_bound, rel_tolerance);
	EXPECT_NEAR(x[3], lower_bound, abs_tolerance);

	abs_tolerance = CMath::get_abs_tolerance(lower_bound, rel_tolerance);
	EXPECT_NEAR(x[4], lower_bound, abs_tolerance);

}

TEST(lbfgs, lbfgs_with_adjust_step_test2)
{
	index_t len=5;
	SGVector<float64_t> x(len);
	Map<VectorXd> eigen_x(x.vector, x.vlen);
	eigen_x.fill(10);
	lbfgs_parameter_t lbfgs_param=init_lbfgs_parameters();
	float64_t opt_value=CMath::INFTY;
	lbfgs(x.vlen, x.vector, &opt_value,
		evaluate_strict_bounded, NULL, NULL, &lbfgs_param, &adjust_step_strict_bounded);

	float64_t rel_tolerance=strict_scale;
	float64_t abs_tolerance;

	abs_tolerance = CMath::get_abs_tolerance(len, rel_tolerance);
	EXPECT_NEAR(opt_value, len, abs_tolerance);

	eigen_x=eigen_x.array()-lower_bound;

	EXPECT_PRED_FORMAT2(::testing::DoubleLE, strict_scale, x[0]);
	EXPECT_PRED_FORMAT2(::testing::DoubleLE, strict_scale, x[1]);
	EXPECT_PRED_FORMAT2(::testing::DoubleLE, strict_scale, x[2]);
	EXPECT_PRED_FORMAT2(::testing::DoubleLE, strict_scale, x[3]);
	EXPECT_PRED_FORMAT2(::testing::DoubleLE, strict_scale, x[4]);
}
