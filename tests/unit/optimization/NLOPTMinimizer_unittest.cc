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
#include "NLOPTMinimizer_unittest.h"
#ifdef USE_GPL_SHOGUN
#ifdef HAVE_NLOPT
#include <gtest/gtest.h>
#include <shogun/lib/Map.h>
#include <shogun/optimization/NLOPTMinimizer.h>
CPiecewiseQuadraticObject2::CPiecewiseQuadraticObject2()
	:SGObject()
{
	init();
}

CPiecewiseQuadraticObject2::~CPiecewiseQuadraticObject2()
{
}

void CPiecewiseQuadraticObject2::init()
{
	m_init_x=SGVector<float64_t>();
	m_truth_x=SGVector<float64_t>();

	SG_ADD(&m_init_x, "init_x", "init_x",
		ParameterProperties::HYPER | ParameterProperties::GRADIENT);

	SG_ADD(&m_truth_x, "truth_x", "truth_x");
}

void CPiecewiseQuadraticObject2::set_init_x(const SGVector<float64_t>& init_x)
{
	m_init_x=init_x;
}

void CPiecewiseQuadraticObject2::set_truth_x(const SGVector<float64_t>& truth_x)
{
	m_truth_x=truth_x;
}

float64_t CPiecewiseQuadraticObject2::get_value()
{
	require(m_init_x.vlen==m_truth_x.vlen, "the length must be the same");
	float64_t res=0.0;
	for(index_t i=0; i<m_init_x.vlen; i++)
	{
		float64_t diff=(m_init_x[i]-m_truth_x[i]);
		res+=diff*diff;
	}
	return res;
}

SGVector<float64_t> CPiecewiseQuadraticObject2::get_gradient(Parameters::const_reference param)
{
	require(param.first == "init_x", "Can't compute derivative wrt {}.{} parameter",
		get_name(), param.first);

	SGVector<float64_t> res;
	if (param.first == "init_x")
	{
		res=SGVector<float64_t>(m_init_x.vlen);
		require(m_init_x.vlen==m_truth_x.vlen, "the length must be the same");
		for(index_t i=0; i<res.vlen; i++)
		{
			float64_t grad=2.0*(m_init_x[i]-m_truth_x[i]);
			res[i]=grad;
		}
	}
	return res;
}
SGVector<float64_t> CPiecewiseQuadraticObject2::get_variable(Parameters::const_reference param)
{
	require(param.first == "init_x", "Can't compute derivative wrt {}.{} parameter",
		get_name(), param.first);

	if (param.first == "init_x")
		return m_init_x;
	return SGVector<float64_t>();
}

SGVector<float64_t> CPiecewiseQuadraticObject2::get_upper_bound(Parameters::const_reference param)
{
	require(param.first == "init_x", "Can't compute derivative wrt {}.{} parameter",
		get_name(), param.first);

	SGVector<float64_t> bound;
	if (param.first == "init_x")
	{
		bound=SGVector<float64_t>(m_init_x.vlen);
		bound.set_const(m_x_upper_bound);
	}
	return bound;
}

SGVector<float64_t> CPiecewiseQuadraticObject2::get_lower_bound(Parameters::const_reference param)
{
	require(param.first == "init_x", "Can't compute derivative wrt {}.{} parameter",
		get_name(), param.first);

	SGVector<float64_t> bound;
	if (param.first == "init_x")
	{
		bound=SGVector<float64_t>(m_init_x.vlen);
		bound.set_const(m_x_lower_bound);
	}
	return bound;
}

NLOPTTestCostFunction::NLOPTTestCostFunction()
	:FirstOrderBoundConstraintsCostFunction()
{
	init();
}

void NLOPTTestCostFunction::init()
{
	m_obj=NULL;
}

NLOPTTestCostFunction::~NLOPTTestCostFunction()
{
	
}

void NLOPTTestCostFunction::set_target(const std::shared_ptr<CPiecewiseQuadraticObject2 >&obj)
{
	if(obj!=m_obj)
	{
		m_obj=obj;
	}
}

float64_t NLOPTTestCostFunction::get_cost()
{
	require(m_obj,"object not set");
	return m_obj->get_value();
}

SGVector<float64_t> NLOPTTestCostFunction::get_lower_bound()
{
	require(m_obj,"object not set");
	std::map<SGObject::Parameters::value_type, std::shared_ptr<SGObject>> parameters;
	m_obj->build_gradient_parameter_dictionary(parameters);
	SGVector<float64_t> bound;

	for (const auto& p: parameters)
		bound=m_obj->get_lower_bound(p.first);
	
	return bound;
}

SGVector<float64_t> NLOPTTestCostFunction::get_upper_bound()
{
	require(m_obj,"object not set");
	std::map<SGObject::Parameters::value_type, std::shared_ptr<SGObject>> parameters;
	m_obj->build_gradient_parameter_dictionary(parameters);
	SGVector<float64_t> bound;

	for (const auto& p: parameters)
	{
		bound=m_obj->get_upper_bound(p.first);
	}
	
	return bound;
}

SGVector<float64_t> NLOPTTestCostFunction::obtain_variable_reference()
{
	require(m_obj,"object not set");
	std::map<SGObject::Parameters::value_type, std::shared_ptr<SGObject>> parameters;
	m_obj->build_gradient_parameter_dictionary(parameters);
	SGVector<float64_t> variable;

	for (const auto& p: parameters)
	{
		variable=m_obj->get_variable(p.first);
	}
	
	return variable;
}

SGVector<float64_t>  NLOPTTestCostFunction::get_gradient()
{
	require(m_obj,"object not set");
	std::map<SGObject::Parameters::value_type, std::shared_ptr<SGObject>> parameters;
	m_obj->build_gradient_parameter_dictionary(parameters);

	SGVector<float64_t> gradient;
	for (const auto& p: parameters)
	{
		gradient=m_obj->get_gradient(p.first);
	}
	
	return gradient;
}

TEST(CNLOPTMinimizer,test1)
{
	auto obj=std::make_shared<CPiecewiseQuadraticObject2>();
	SGVector<float64_t> init_x(5);
	init_x.set_const(0.0);
	SGVector<float64_t> truth_x(5);
	for(index_t idx=0; idx<truth_x.vlen; idx++)
	{
		if(idx<2)
			truth_x[idx]=5.0;
		else
			truth_x[idx]=3.0;

	}
	obj->set_init_x(init_x);
	obj->set_truth_x(truth_x);

	obj->set_x_lower_bound(0.0);
	obj->set_x_upper_bound(4.0);

	auto b=std::make_shared<NLOPTTestCostFunction>();
	
	b->set_target(obj);
	
	auto opt=std::make_shared<NLOPTMinimizer>(b);
	float64_t cost=opt->minimize();
	EXPECT_NEAR(cost, 2.0, 1e-6);
}

TEST(CNLOPTMinimizer,test2)
{
	auto obj=std::make_shared<CPiecewiseQuadraticObject2>();
	SGVector<float64_t> init_x(5);
	init_x.set_const(0.0);
	SGVector<float64_t> truth_x(5);
	for(index_t idx=0; idx<truth_x.vlen; idx++)
	{
		if(idx<2)
			truth_x[idx]=5.0;
		else
			truth_x[idx]=3.0;

	}
	obj->set_init_x(init_x);
	obj->set_truth_x(truth_x);

	obj->set_x_lower_bound(0.0);
	obj->set_x_upper_bound(4.0);

	auto b=std::make_shared<NLOPTTestCostFunction>();
	
	b->set_target(obj);
	
	auto opt=std::make_shared<NLOPTMinimizer>(b);
	opt->minimize();

	for(index_t i=0; i<init_x.vlen; i++)
	{
		if(i<2)
		{
			EXPECT_NEAR(init_x[i], 4.0, 1e-6);
		}
		else
		{
			EXPECT_NEAR(init_x[i], 3.0, 1e-6);
		}
	}
}
#endif /* HAVE_NLOPT */
#endif //USE_GPL_SHOGUN
