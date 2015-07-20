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
#include <gtest/gtest.h>
#include "NLOPTMinimizer_unittest.h"
CPiecewiseQuadraticObject2::CPiecewiseQuadraticObject2()
	:CSGObject()
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
	m_init_y=SGVector<float64_t>();
	m_truth_y=SGVector<float64_t>();

	SG_ADD(&m_init_x, "init_x", "init_x",
		MS_AVAILABLE, GRADIENT_AVAILABLE);
	SG_ADD(&m_init_y, "init_y", "init_y",
		MS_AVAILABLE, GRADIENT_AVAILABLE);

	SG_ADD(&m_truth_x, "truth_x", "truth_x",
		MS_NOT_AVAILABLE);
	SG_ADD(&m_truth_y, "truth_y", "truth_y",
		MS_NOT_AVAILABLE);
}

void CPiecewiseQuadraticObject2::set_init_x(SGVector<float64_t> init_x)
{
	m_init_x=init_x;
}

void CPiecewiseQuadraticObject2::set_truth_x(SGVector<float64_t> truth_x)
{
	m_truth_x=truth_x;
}

void CPiecewiseQuadraticObject2::set_init_y(SGVector<float64_t> init_y)
{
	m_init_y=init_y;
}

void CPiecewiseQuadraticObject2::set_truth_y(SGVector<float64_t> truth_y)
{
	m_truth_y=truth_y;
}

float64_t CPiecewiseQuadraticObject2::get_value()
{
	REQUIRE(m_init_x.vlen==m_truth_x.vlen, "the length must be the same\n");
	REQUIRE(m_init_y.vlen==m_truth_y.vlen, "the length must be the same\n");
	float64_t res=0.0;
	for(index_t i=0; i<m_init_x.vlen; i++)
	{
		float64_t diff=(m_init_x[i]-m_truth_x[i]);
		res+=diff*diff;
	}
	for(index_t i=0; i<m_init_y.vlen; i++)
	{
		float64_t diff=(m_init_y[i]-m_truth_y[i]);
		res+=diff*diff;
	}
	return res;
}

SGVector<float64_t> CPiecewiseQuadraticObject2::get_gradient(TParameter * param)
{
	REQUIRE(param, "param not set\n");
	REQUIRE(!strcmp(param->m_name, "init_x") || !strcmp(param->m_name, "init_y"), "Can't compute derivative wrt %s.%s parameter\n",
		get_name(), param->m_name);

	SGVector<float64_t> res;
	if (!strcmp(param->m_name, "init_x"))
	{
		res=SGVector<float64_t>(m_init_x.vlen);
		REQUIRE(m_init_x.vlen==m_truth_x.vlen, "the length must be the same\n");
		for(index_t i=0; i<res.vlen; i++)
		{
			float64_t grad=2.0*(m_init_x[i]-m_truth_x[i]);
			res[i]=grad;
		}
	}
	else if(!strcmp(param->m_name, "init_y"))
	{
		res=SGVector<float64_t>(m_init_y.vlen);
		REQUIRE(m_init_y.vlen==m_truth_y.vlen, "the length must be the same\n");
		for(index_t i=0; i<res.vlen; i++)
		{
			float64_t grad=2.0*(m_init_y[i]-m_truth_y[i]);
			res[i]=grad;
		}
	}

	return res;
}
SGVector<float64_t> CPiecewiseQuadraticObject2::get_variable(TParameter * param)
{
	REQUIRE(param, "param not set\n");

	REQUIRE(!strcmp(param->m_name, "init_x") || !strcmp(param->m_name, "init_y"), "Can't compute derivative wrt %s.%s parameter\n",
		get_name(), param->m_name);

	if (!strcmp(param->m_name, "init_x"))
		return m_init_x;
	if(!strcmp(param->m_name, "init_y"))
		return m_init_y;
	return SGVector<float64_t>();
}

SGVector<float64_t> CPiecewiseQuadraticObject2::get_upper_bound(TParameter * param)
{
	REQUIRE(param, "param not set\n");

	REQUIRE(!strcmp(param->m_name, "init_x") || !strcmp(param->m_name, "init_y"), "Can't compute derivative wrt %s.%s parameter\n",
		get_name(), param->m_name);

	SGVector<float64_t> bound;
	if (!strcmp(param->m_name, "init_x"))
	{
		bound=SGVector<float64_t>(m_init_x.vlen);
		bound.set_const(m_x_upper_bound);
	}
	if(!strcmp(param->m_name, "init_y"))
	{
		bound=SGVector<float64_t>(m_init_y.vlen);
		bound.set_const(m_y_upper_bound);
	}
	return bound;
}

SGVector<float64_t> CPiecewiseQuadraticObject2::get_lower_bound(TParameter * param)
{
	REQUIRE(param, "param not set\n");

	REQUIRE(!strcmp(param->m_name, "init_x") || !strcmp(param->m_name, "init_y"), "Can't compute derivative wrt %s.%s parameter\n",
		get_name(), param->m_name);

	SGVector<float64_t> bound;
	if (!strcmp(param->m_name, "init_x"))
	{
		bound=SGVector<float64_t>(m_init_x.vlen);
		bound.set_const(m_x_lower_bound);
	}
	if(!strcmp(param->m_name, "init_y"))
	{
		bound=SGVector<float64_t>(m_init_y.vlen);
		bound.set_const(m_y_lower_bound);
	}
	return bound;
}


CMyCostFunction2::CMyCostFunction2()
	:CFirstOrderBoundConstraintsCostFunction()
{
	init();
}
void CMyCostFunction2::init()
{
	m_obj=NULL;
}
CMyCostFunction2::~CMyCostFunction2()
{
	SG_UNREF(m_obj);
}
void CMyCostFunction2::set_target(CPiecewiseQuadraticObject2 *obj)
{
	if(obj!=m_obj)
	{
		SG_UNREF(m_obj);
		m_obj=obj;
		SG_REF(m_obj);
	}
}

float64_t CMyCostFunction2::get_cost()
{
	REQUIRE(m_obj,"object not set\n");
	return m_obj->get_value();
}

CMap<TParameter*, SGVector<float64_t> >* CMyCostFunction2::get_lower_bound()
{
	REQUIRE(m_obj,"object not set\n");
	CMap<TParameter*, CSGObject*>* parameters=new CMap<TParameter*, CSGObject*>();
	m_obj->build_gradient_parameter_dictionary(parameters);
	index_t num_variables=parameters->get_num_elements();
	CMap<TParameter*, SGVector<float64_t> >* bounds=
		new CMap<TParameter*, SGVector<float64_t> >(num_variables, num_variables);

	for(index_t i=0; i<num_variables; i++)
	{
		CMapNode<TParameter*, CSGObject*>* node=parameters->get_node_ptr(i);
		if(node->data==m_obj)
		{
			SGVector<float64_t> bound=m_obj->get_lower_bound(node->key);
			bounds->add(node->key,bound);
			std::cout<<node->key->m_name<<std::endl;
			std::cout<<"lower bound"<<std::endl;
			for(int j=0;j<bound.vlen;j++)
				std::cout<<bound[j]<<" ";
			std::cout<<std::endl;
		}
	}
	SG_UNREF(parameters);
	return bounds;
}

CMap<TParameter*, SGVector<float64_t> >* CMyCostFunction2::get_upper_bound()
{
	REQUIRE(m_obj,"object not set\n");
	CMap<TParameter*, CSGObject*>* parameters=new CMap<TParameter*, CSGObject*>();
	m_obj->build_gradient_parameter_dictionary(parameters);
	index_t num_variables=parameters->get_num_elements();
	CMap<TParameter*, SGVector<float64_t> >* bounds=
		new CMap<TParameter*, SGVector<float64_t> >(num_variables, num_variables);

	for(index_t i=0; i<num_variables; i++)
	{
		CMapNode<TParameter*, CSGObject*>* node=parameters->get_node_ptr(i);
		if(node->data==m_obj)
		{
			SGVector<float64_t> bound=m_obj->get_upper_bound(node->key);
			bounds->add(node->key,bound);
			std::cout<<node->key->m_name<<std::endl;
			std::cout<<"upper bound"<<std::endl;
			for(int j=0;j<bound.vlen;j++)
				std::cout<<bound[j]<<" ";
			std::cout<<std::endl;
		}
	}
	SG_UNREF(parameters);
	return bounds;
}

CMap<TParameter*, SGVector<float64_t> >* CMyCostFunction2::obtain_variable_reference()
{
	REQUIRE(m_obj,"object not set\n");
	CMap<TParameter*, CSGObject*>* parameters=new CMap<TParameter*, CSGObject*>();
	m_obj->build_gradient_parameter_dictionary(parameters);
	index_t num_variables=parameters->get_num_elements();
	CMap<TParameter*, SGVector<float64_t> >* variables=
		new CMap<TParameter*, SGVector<float64_t> >(num_variables, num_variables);

	for(index_t i=0; i<num_variables; i++)
	{
		CMapNode<TParameter*, CSGObject*>* node=parameters->get_node_ptr(i);
		if(node->data==m_obj)
		{
			SGVector<float64_t> var=m_obj->get_variable(node->key);
			variables->add(node->key,var);
			std::cout<<node->key->m_name<<std::endl;
			for(int j=0;j<var.vlen;j++)
				std::cout<<var[j]<<" ";
			std::cout<<std::endl;
		}
	}

	SG_UNREF(parameters);
	return variables;
}
CMap<TParameter*, SGVector<float64_t> >* CMyCostFunction2::get_gradient()
{
	REQUIRE(m_obj,"object not set\n");
	CMap<TParameter*, CSGObject*>* parameters=new CMap<TParameter*, CSGObject*>();
	m_obj->build_gradient_parameter_dictionary(parameters);

	index_t num_gradients=parameters->get_num_elements();
	CMap<TParameter*, SGVector<float64_t> >* gradients=
		new CMap<TParameter*, SGVector<float64_t> >(num_gradients, num_gradients);

	for(index_t i=0; i<num_gradients; i++)
	{
		CMapNode<TParameter*, CSGObject*>* node=parameters->get_node_ptr(i);
		if(node->data==m_obj)
		{
			SGVector<float64_t> grad=m_obj->get_gradient(node->key);
			gradients->add(node->key,grad);
			std::cout<<node->key->m_name<<std::endl;
			for(int j=0;j<grad.vlen;j++)
				std::cout<<grad[j]<<" ";
			std::cout<<std::endl;
		}
	}
	SG_UNREF(parameters);
	return gradients;
}

TEST(NLOPTMinimizer,test1)
{
	CPiecewiseQuadraticObject2 *obj=new CPiecewiseQuadraticObject2();
	SGVector<float64_t> init_x(5);
	init_x.set_const(0.0);
	SGVector<float64_t> truth_x(5);
	truth_x.set_const(5.0);
	obj->set_init_x(init_x);
	obj->set_truth_x(truth_x);

	SGVector<float64_t> init_y(2);
	init_y.set_const(0.0);
	SGVector<float64_t> truth_y(2);
	truth_y.set_const(3.0);
	obj->set_init_y(init_y);
	obj->set_truth_y(truth_y);

	obj->set_x_lower_bound(0.0);
	obj->set_y_lower_bound(0.0);
	obj->set_x_upper_bound(4.0);
	obj->set_y_upper_bound(4.0);

	CMyCostFunction2 *b=new CMyCostFunction2();
	SG_REF(obj);
	b->set_target(obj);
	
	SG_REF(b);
	CFirstOrderMinimizer* opt=new CNLOPTMinimizer(b);
	float64_t cost=opt->minimization();
	EXPECT_NEAR(cost, 5.0, 1e-6);

	SG_UNREF(obj);
	SG_UNREF(b);
	SG_UNREF(opt);
}

TEST(NLOPTMinimizer,test2)
{
	CPiecewiseQuadraticObject2 *obj=new CPiecewiseQuadraticObject2();
	SGVector<float64_t> init_x(5);
	init_x.set_const(0.0);
	SGVector<float64_t> truth_x(5);
	truth_x.set_const(5.0);
	obj->set_init_x(init_x);
	obj->set_truth_x(truth_x);

	SGVector<float64_t> init_y(2);
	init_y.set_const(0.0);
	SGVector<float64_t> truth_y(2);
	truth_y.set_const(3.0);
	obj->set_init_y(init_y);
	obj->set_truth_y(truth_y);

	obj->set_x_lower_bound(0.0);
	obj->set_y_lower_bound(0.0);
	obj->set_x_upper_bound(4.0);
	obj->set_y_upper_bound(4.0);

	CMyCostFunction2 *b=new CMyCostFunction2();
	SG_REF(obj);
	b->set_target(obj);
	
	SG_REF(b);
	CFirstOrderMinimizer* opt=new CNLOPTMinimizer(b);
	opt->minimization();

	for(index_t i=0; i<init_x.vlen; i++)
	{
		EXPECT_NEAR(init_x[i], 4.0, 1e-6);
	}

	for(index_t i=0; i<init_y.vlen; i++)
	{
		EXPECT_NEAR(init_y[i], 3.0, 1e-6);
	}

	SG_UNREF(obj);
	SG_UNREF(b);
	SG_UNREF(opt);
}
