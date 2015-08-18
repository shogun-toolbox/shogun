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
#include <shogun/lib/config.h>
#ifdef HAVE_EIGEN3
#ifndef SGDMINIMIZER_UNITTEST_H
#define SGDMINIMIZER_UNITTEST_H
#include <shogun/optimization/FirstOrderStochasticCostFunction.h>
#include <shogun/lib/SGMatrix.h>
#include <shogun/lib/SGVector.h>
#include <shogun/base/SGObject.h>
using namespace shogun;

class CRegressionExample;

class SGDTestStochasticCostFunction: public FirstOrderStochasticCostFunction
{
public:
	SGDTestStochasticCostFunction();
	virtual ~SGDTestStochasticCostFunction();
	void set_target(CRegressionExample *obj);
	virtual float64_t get_cost();
	virtual SGVector<float64_t> obtain_variable_reference();
	virtual SGVector<float64_t> get_gradient();
	virtual void begin_sample();
	virtual bool next_sample();
private:
	index_t m_idx;
	void init();
	CRegressionExample* m_obj;
};

class CRegressionExample: public CSGObject
{
friend class SGDTestStochasticCostFunction;
public:
	CRegressionExample();
	virtual ~CRegressionExample(){}
	float64_t get_cost();
	virtual const char* get_name() const {return "RegressionExample";}
	void set_x(SGMatrix<float64_t> x){m_x=x;}
	void set_y(SGVector<float64_t> y){m_y=y;}
	void set_init_w(SGVector<float64_t> w){m_w=w;}
private:
	int get_sample_size();
	SGVector<float64_t> get_gradient(TParameter * param, index_t idx);
	SGVector<float64_t> get_variable(TParameter * param);
	SGMatrix<float64_t> m_x;
	SGVector<float64_t> m_w;
	SGVector<float64_t> m_y;
	void init();
};
#endif
#endif /*  HAVE_EIGEN3 */
