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

#ifndef STOCHASTICMINIMIZERS_UNITTEST_H
#define STOCHASTICMINIMIZERS_UNITTEST_H
#include <shogun/optimization/FirstOrderSAGCostFunction.h>
#include <shogun/lib/SGMatrix.h>
#include <shogun/lib/SGVector.h>
#include <shogun/base/SGObject.h>
using namespace shogun;

class CRegressionExample;

class RegressionForTestCostFunction: public FirstOrderSAGCostFunction
{
public:
	RegressionForTestCostFunction();
	virtual ~RegressionForTestCostFunction();
	void set_target(CRegressionExample *obj);
	virtual float64_t get_cost();
	virtual SGVector<float64_t> obtain_variable_reference();
	virtual SGVector<float64_t> get_gradient();
	virtual SGVector<float64_t> get_average_gradient();
	virtual int32_t get_sample_size();
	virtual void begin_sample();
	virtual bool next_sample();
private:
	index_t m_idx;
	void init();
	CRegressionExample* m_obj;
};

class ClassificationForTestCostFunction: public FirstOrderSAGCostFunction
{
public:
	ClassificationForTestCostFunction();
	virtual ~ClassificationForTestCostFunction(){}
	virtual void set_data(SGMatrix<float64_t> features, SGVector<float64_t> labels);
	virtual float64_t get_cost();
	virtual void set_sample_sequences(SGVector<int32_t> index, index_t num_sequences);
	virtual SGVector<float64_t> obtain_variable_reference();
	virtual SGVector<float64_t> get_gradient();
	virtual void begin_sample();
	virtual bool next_sample();
	virtual int32_t get_sample_size();
	virtual SGVector<float64_t> get_average_gradient();

protected:
	index_t m_sample_idx;
	SGVector<int32_t> m_sample_sequences;
	index_t m_num_sequences;
	bool is_begin;
	SGVector<float64_t> m_labels;
	SGMatrix<float64_t> m_features;
	SGVector<float64_t> m_weight;
	index_t m_call_times;
	void init();
};

class ClassificationForTestCostFunction2: public ClassificationForTestCostFunction
{
public:
	ClassificationForTestCostFunction2()
		:ClassificationForTestCostFunction(){};
	virtual ~ClassificationForTestCostFunction2(){};
	virtual void begin_sample();
	virtual bool next_sample();
};

class CRegressionExample: public CSGObject
{
friend class RegressionForTestCostFunction;
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
	SGVector<float64_t> get_gradient(TParameter * param);
	SGVector<float64_t> get_variable(TParameter * param);
	SGMatrix<float64_t> m_x;
	SGVector<float64_t> m_w;
	SGVector<float64_t> m_y;
	void init();
};
#endif
