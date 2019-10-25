/*
 * Copyright (c) The Shogun Machine Learning Toolbox
 * Written (W) 2015 Wu Lin
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
 */
#ifndef LBFGSMINIMIZER_UNITTEST_H
#define LBFGSMINIMIZER_UNITTEST_H
#include <shogun/lib/config.h>
#include <shogun/optimization/FirstOrderCostFunction.h>
#include <shogun/optimization/lbfgs/LBFGSMinimizer.h>
using namespace shogun;
class PiecewiseQuadraticObject;

class LBFGSTestCostFunction: public FirstOrderCostFunction
{
public:
	LBFGSTestCostFunction();
	virtual ~LBFGSTestCostFunction();
	void set_target(std::shared_ptr<PiecewiseQuadraticObject> obj);
	virtual float64_t get_cost();
	virtual SGVector<float64_t> obtain_variable_reference();
	virtual SGVector<float64_t> get_gradient();
	virtual const char* get_name() const { return "LBFGSTestCostFunction"; }
private:
	void init();
	std::shared_ptr<PiecewiseQuadraticObject> m_obj;
};

class PiecewiseQuadraticObject: public SGObject
{
friend class LBFGSTestCostFunction;
public:
	PiecewiseQuadraticObject();
	virtual ~PiecewiseQuadraticObject();
	void set_init_x(SGVector<float64_t> init_x);
	void set_truth_x(SGVector<float64_t> truth_x);
	float64_t get_value();
	virtual const char* get_name() const {return "PiecewiseQuadraticObject";}
private:
	SGVector<float64_t> get_gradient(Parameters::const_reference param);
	SGVector<float64_t> get_variable(Parameters::const_reference param);
	void init();
	SGVector<float64_t> m_init_x;
	SGVector<float64_t> m_truth_x;
};
#endif
