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
#ifndef NLOPTMINIMIZER_UNITTEST_H
#define NLOPTMINIMIZER_UNITTEST_H
#include <shogun/lib/config.h>
#include <shogun/optimization/FirstOrderBoundConstraintsCostFunction.h>
#include <shogun/optimization/NLOPTMinimizer.h>
using namespace shogun;
class CPiecewiseQuadraticObject2;

class CMyCostFunction2: public CFirstOrderBoundConstraintsCostFunction
{
public:
	CMyCostFunction2();
	virtual ~CMyCostFunction2();
	void set_target(CPiecewiseQuadraticObject2 *obj);
	virtual float64_t get_cost();
	virtual CMap<TParameter*, SGVector<float64_t> >* obtain_variable_reference();
	virtual CMap<TParameter*, SGVector<float64_t> >* get_gradient();
	virtual CMap<TParameter*, SGVector<float64_t> >* get_lower_bound();
	virtual CMap<TParameter*, SGVector<float64_t> >* get_upper_bound();
	virtual const char* get_name() const {return "MyCostFunction2";}
private:
	void init();
	CPiecewiseQuadraticObject2 *m_obj;
};


class CPiecewiseQuadraticObject2: public CSGObject
{
friend class CMyCostFunction2;
public:
	CPiecewiseQuadraticObject2();
	virtual ~CPiecewiseQuadraticObject2();
	void set_init_x(SGVector<float64_t> init_x);
	void set_truth_x(SGVector<float64_t> truth_x);
	void set_init_y(SGVector<float64_t> init_y);
	void set_truth_y(SGVector<float64_t> truth_y);

	void set_x_lower_bound(float64_t lower_bound){m_x_lower_bound=lower_bound;}
	void set_y_lower_bound(float64_t lower_bound){m_y_lower_bound=lower_bound;}
	void set_x_upper_bound(float64_t upper_bound){m_x_upper_bound=upper_bound;}
	void set_y_upper_bound(float64_t upper_bound){m_y_upper_bound=upper_bound;}

	float64_t get_value();
	virtual const char* get_name() const {return "PiecewiseQuadraticObject";}
private:
	SGVector<float64_t> get_gradient(TParameter * param);
	SGVector<float64_t> get_variable(TParameter * param);
	SGVector<float64_t> get_upper_bound(TParameter * param);
	SGVector<float64_t> get_lower_bound(TParameter * param);
	void init();
	SGVector<float64_t> m_init_x;
	SGVector<float64_t> m_truth_x;
	SGVector<float64_t> m_init_y;
	SGVector<float64_t> m_truth_y;
	float64_t m_x_lower_bound;
	float64_t m_x_upper_bound;
	float64_t m_y_lower_bound;
	float64_t m_y_upper_bound;
};

#endif
