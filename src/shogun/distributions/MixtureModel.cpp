/*
 * Copyright (c) The Shogun Machine Learning Toolbox
 * Written (w) 2014 Parijat Mazumdar
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice, this
 * list of conditions and the following disclaimer.
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 * this list of conditions and the following disclaimer in the documentation
 * and/or other materials provided with the distribution.
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
 */

#include <shogun/distributions/MixtureModel.h>
#include <shogun/mathematics/Math.h> 

using namespace shogun;

CMixtureModel::CMixtureModel()
{
	init();
}

CMixtureModel::CMixtureModel(CDynamicObjectArray* components, SGVector<float64_t> weights)
{
	init();
	m_components=components;
	SG_REF(components);
	m_weights=weights;
}

CMixtureModel::~CMixtureModel()
{
	SG_UNREF(m_components);
}

bool CMixtureModel::train(CFeatures* data)
{
	//TBD
	return true;
}

float64_t CMixtureModel::get_log_model_parameter(int32_t num_param)
{
	REQUIRE(num_param==1,"number of parameters in mixture model is 1"
	" (i.e. number of components). num_components should be 1. %d supplied\n",num_param)

	return CMath::log(get_num_components());
}

float64_t CMixtureModel::get_log_derivative(int32_t num_param, int32_t num_example)
{
	// TBD
	return 0;
}

float64_t CMixtureModel::get_log_likelihood_example(int32_t num_example)
{
	// TBD
	return 0;	
}

SGVector<float64_t> CMixtureModel::get_weights() const
{
	return m_weights;
}

void CMixtureModel::set_weights(SGVector<float64_t> weights)
{
	m_weights=weights;
}

CDynamicObjectArray* CMixtureModel::get_components() const
{
	SG_REF(m_components);
	return m_components;
}

void CMixtureModel::set_components(CDynamicObjectArray* components)
{
	if (m_components!=NULL)
		SG_UNREF(m_components)

	m_components=components;
	SG_REF(m_components);
}

index_t CMixtureModel::get_num_components() const
{
	return m_components->get_num_elements();
}

CDistribution* CMixtureModel::get_component(index_t index) const
{
	REQUIRE(index<get_num_components(),"index supplied (%d) is greater than total mixture components (%d)\n"
																				,index,get_num_components())
	return CDistribution::obtain_from_generic(m_components->get_element(index));
}

SGVector<float64_t> CMixtureModel::sample()
{
	// TBD
	return SGVector<float64_t>();
}

SGVector<float64_t> CMixtureModel::cluster(SGVector<float64_t> point)
{
	// TBD
	return point;
}

void CMixtureModel::init()
{
	m_components=NULL;
	m_weights=SGVector<float64_t>();

	SG_ADD((CSGObject**)&m_components,"m_components","components of mixture",MS_NOT_AVAILABLE);
	SG_ADD(&m_weights,"m_weights","weights of components",MS_NOT_AVAILABLE);	
}