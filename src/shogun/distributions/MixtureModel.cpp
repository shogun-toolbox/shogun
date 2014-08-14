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
#include <shogun/features/DotFeatures.h>
#include <shogun/features/DenseFeatures.h>
#include <shogun/distributions/EMMixtureModel.h>

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
	REQUIRE(m_components->get_num_elements()>0,"mixture componenents not specified\n")
	REQUIRE(m_components->get_num_elements()==m_weights.vlen,"number of weights (%d) does  not"
		" match number of components (%d)\n",m_weights.vlen,m_components->get_num_elements())

	// set training features
	if (data)
	{
		if (!data->has_property(FP_DOT))
				SG_ERROR("Specified features are not of type CDotFeatures\n")
		set_features(data);
	}
	else if (!features)
	{
		SG_ERROR("No features to train on.\n")
	}

	// set training points in all components of the mixture 
	for (int32_t i=0;i<m_components->get_num_elements();i++)
	{
		CDistribution* comp=CDistribution::obtain_from_generic(m_components->get_element(i));
		comp->set_features(features);

		SG_UNREF(comp)
	}

	CDotFeatures* dotdata=dynamic_cast<CDotFeatures *>(features);
	REQUIRE(dotdata,"dynamic cast from CFeatures to CDotFeatures returned NULL")
	int32_t num_vectors=dotdata->get_num_vectors();

	// set data for EM
	CEMMixtureModel* em=new CEMMixtureModel();
	em->data.alpha=SGMatrix<float64_t>(num_vectors,m_components->get_num_elements());
	em->data.components=m_components;
	em->data.weights=m_weights;

	// run EM
	bool is_converged=em->iterate_em(m_max_iters,m_conv_tol);
	if (!is_converged)
		SG_WARNING("max iterations reached. No convergence yet!\n")

	SG_UNREF(em)
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
	SG_NOTIMPLEMENTED
	return 0;
}

float64_t CMixtureModel::get_log_likelihood_example(int32_t num_example)
{
	REQUIRE(features,"features not set\n")
	REQUIRE(features->get_feature_class() == C_DENSE,"Dense features required\n")
	REQUIRE(features->get_feature_type() == F_DREAL,"Real features required\n")

	SGVector<float64_t> log_likelihood_component(m_components->get_num_elements());
	for (int32_t i=0;i<m_components->get_num_elements();i++)
	{
		CDistribution* ith_comp=CDistribution::obtain_from_generic(m_components->get_element(i));
		log_likelihood_component[i]=ith_comp->get_log_likelihood_example(num_example)+CMath::log(m_weights[i]);

		SG_UNREF(ith_comp);
	}

	return CMath::log_sum_exp(log_likelihood_component);
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

void CMixtureModel::set_max_iters(int32_t max_iters)
{
	m_max_iters=max_iters;
}

int32_t CMixtureModel::get_max_iters() const
{
	return m_max_iters;
}

void CMixtureModel::set_convergence_tolerance(float64_t conv_tol)
{
	m_conv_tol=conv_tol;
}

float64_t CMixtureModel::get_convergence_tolerance() const
{
	return m_conv_tol;
}

SGVector<float64_t> CMixtureModel::sample()
{
	// TBD
	SG_NOTIMPLEMENTED;
	return SGVector<float64_t>();
}

SGVector<float64_t> CMixtureModel::cluster(SGVector<float64_t> point)
{
	// TBD
	SG_NOTIMPLEMENTED;
	return point;
}

void CMixtureModel::init()
{
	m_components=NULL;
	m_weights=SGVector<float64_t>();
	m_conv_tol=1e-8;
	m_max_iters=1000;

	SG_ADD((CSGObject**)&m_components,"m_components","components of mixture",MS_NOT_AVAILABLE);
	SG_ADD(&m_weights,"m_weights","weights of components",MS_NOT_AVAILABLE);
	SG_ADD(&m_conv_tol,"m_conv_tol","convergence tolerance",MS_NOT_AVAILABLE);
	SG_ADD(&m_max_iters,"m_max_iters","max number of iterations",MS_NOT_AVAILABLE);
}