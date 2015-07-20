 /*
 * Copyright (c) The Shogun Machine Learning Toolbox
 * Written (w) 2015 Wu Lin
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
 * This code specifically adapted from function in approxKL.m and infKL.m
 */
#include <shogun/lib/config.h>
#include <shogun/optimization/WrappedMinimizer.h>
#include <algorithm>
#include <shogun/base/Parameter.h>

using namespace shogun;

CWrappedMinimizer::CWrappedMinimizer()
	:CFirstOrderMinimizer()
{
	init();
}

CWrappedMinimizer::~CWrappedMinimizer()
{
	SG_UNREF(m_variable);
}

CWrappedMinimizer::CWrappedMinimizer(CFirstOrderCostFunction *fun)
	:CFirstOrderMinimizer(fun)
{
	init();
}

void CWrappedMinimizer::init()
{
	m_variable_vec=SGVector<float64_t>();
	m_variable=NULL;
	m_is_in_place=false;

	SG_ADD(&m_variable_vec,"variable_vec","variable_vec", MS_NOT_AVAILABLE);
	SG_ADD(&m_is_in_place,"is_in_place","is_in_place", MS_NOT_AVAILABLE);
	SG_ADD((CSGObject **)&m_variable,"m_variable","m_variable", MS_NOT_AVAILABLE);
}

SGVector<float64_t> CWrappedMinimizer::convert(CMap<TParameter*, SGVector<float64_t> >* input)
{
	REQUIRE(input,"Input not set\n");

	int32_t length=0;
	for(int32_t i=0; i<input->get_num_elements(); i++)
	{
		CMapNode<TParameter*, SGVector<float64_t> >* var=input->get_node_ptr(i);
		REQUIRE(var->key->m_name, "The name of variable not set \n");
		REQUIRE((var->data).vlen>0,
			"The length (%d) of variable (%s) is empty \n",
			(var->data).vlen, var->key->m_name);
		length+=(var->data).vlen;
	}
	SGVector<float64_t> output (length);
	int32_t offset=0;
	if (length>0)
	{
		for(int32_t i=0; i<input->get_num_elements(); i++)
		{
			CMapNode<TParameter*, SGVector<float64_t> >* var=input->get_node_ptr(i);
			int32_t vlen=(var->data).vlen;
			std::copy((var->data).vector, (var->data).vector+vlen, output.vector+offset);
			offset+=vlen;
		}
	}
	ASSERT(offset==length);
	return output;
}
void CWrappedMinimizer::copy_in_parameter_order(const float64_t* input, const int32_t dim,
	CMap<TParameter*, SGVector<float64_t> >* parameter_order,
	CMap<TParameter*, SGVector<float64_t> >* output)
{
	REQUIRE(input,"Input not set\n");
	REQUIRE(output,"Output not set\n");
	REQUIRE(parameter_order,"Parameter_order not set\n");
	REQUIRE(dim>0,"The length of input (%d) is not positive\n",dim);

	REQUIRE(output->get_num_elements()==parameter_order->get_num_elements(),
		"The length of output (%d) and the length of parameter_order (%d) do not match\n",
		output->get_num_elements(),parameter_order->get_num_elements());

	int32_t offset=0;
	for(int32_t i=0; i<parameter_order->get_num_elements(); i++)
	{
		CMapNode<TParameter*, SGVector<float64_t> >* var=parameter_order->get_node_ptr(i);

		int32_t j=0;
		for(; j<output->get_num_elements(); j++)
		{
			CMapNode<TParameter*, SGVector<float64_t> >* g=output->get_node_ptr(j);
			REQUIRE(g->key->m_name, "The name of parameter not set \n");
			if (strcmp(var->key->m_name,g->key->m_name)==0)
			{
				int32_t vlen=(g->data).vlen;
				REQUIRE((var->data).vlen==(g->data).vlen,
					"The dim (%d) of parameter %s and the length of its output (%d) do not match\n",
					(var->data).vlen,var->key->m_name,(g->data).vlen);
				REQUIRE(offset+vlen<=dim,"")
				std::copy(input+offset, input+offset+vlen,(g->data).vector);
				offset+=vlen;
				break;
			}
		}
		REQUIRE(j<output->get_num_elements(), "Cannot find %s in output \n",
			var->key->m_name);
	}
	ASSERT(offset==dim);
}

void CWrappedMinimizer::copy_in_parameter_order(CMap<TParameter*, SGVector<float64_t> >* input,
	CMap<TParameter*, SGVector<float64_t> >* parameter_order, float64_t* output, const int32_t dim)
{
	REQUIRE(input,"Input not set\n");
	REQUIRE(output,"Output not set\n");
	REQUIRE(parameter_order,"Parameter_order not set\n");
	REQUIRE(dim>0,"The length of output (%d) is not positive\n",dim);

	REQUIRE(input->get_num_elements()==parameter_order->get_num_elements(),
		"The length of input (%d) and the length of parameter_order (%d) do not match\n",
		input->get_num_elements(),parameter_order->get_num_elements());

	int32_t offset=0;
	for(int32_t i=0; i<parameter_order->get_num_elements(); i++)
	{
		CMapNode<TParameter*, SGVector<float64_t> >* var=parameter_order->get_node_ptr(i);

		int32_t j=0;
		for(; j<input->get_num_elements(); j++)
		{
			CMapNode<TParameter*, SGVector<float64_t> >* g=input->get_node_ptr(j);
			REQUIRE(g->key->m_name, "The name of parameter not set \n");
			if (strcmp(var->key->m_name,g->key->m_name)==0)
			{
				int32_t vlen=(g->data).vlen;
				REQUIRE((var->data).vlen==(g->data).vlen,
					"The dim (%d) of parameter %s and the length of its input (%d) do not match\n",
					(var->data).vlen,var->key->m_name,(g->data).vlen);
				REQUIRE(offset+vlen<=dim,"")
				std::copy((g->data).vector,(g->data).vector+vlen,output+offset);
				offset+=vlen;
				break;
			}
		}
		REQUIRE(j<input->get_num_elements(), "Cannot find %s in input \n",
			var->key->m_name);
	}
	ASSERT(offset==dim);
}

void CWrappedMinimizer::minimization_init()
{
	REQUIRE(m_fun, "Cost function not set!\n");

	CMap<TParameter*, SGVector<float64_t> >* variable=m_fun->obtain_variable_reference();
	REQUIRE(variable,"Target variable from cost function not set!\n");
	REQUIRE(variable->get_num_elements()>0,"Target variable from cost function is empty!\n");
	if(variable!=m_variable)
	{
		SG_UNREF(m_variable);
		m_variable=variable;
		SG_REF(m_variable);
	}

	if(m_variable->get_num_elements()==1)
	{
		CMapNode<TParameter*, SGVector<float64_t> >* var=m_variable->get_node_ptr(0);
		REQUIRE(var->key->m_name, "The name of variable not set \n");
		m_variable_vec=var->data;
		REQUIRE((var->data).vlen>0,
			"The length (%d) of variable (%s) is empty \n",
			(var->data).vlen, var->key->m_name);
		m_is_in_place=true;
	}
	else
	{
		m_variable_vec=convert(m_variable);
		m_is_in_place=false;
	}
}
