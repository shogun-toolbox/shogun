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

MixtureModel::MixtureModel()
{
	init();
}

MixtureModel::MixtureModel(std::shared_ptr<DynamicObjectArray> components, SGVector<float64_t> weights)
{
	init();
	m_components=components;

	m_weights=weights;
}

MixtureModel::~MixtureModel()
{

}

bool MixtureModel::train(std::shared_ptr<Features> data)
{
	require(m_components->get_num_elements()>0,"mixture componenents not specified");
	require(m_components->get_num_elements()==m_weights.vlen,"number of weights ({}) does  not"
		" match number of components ({})",m_weights.vlen,m_components->get_num_elements());

	// set training features
	if (data)
	{
		if (!data->has_property(FP_DOT))
				error("Specified features are not of type CDotFeatures");
		set_features(data);
	}
	else if (!features)
	{
		error("No features to train on.");
	}

	// set training points in all components of the mixture
	for (int32_t i=0;i<m_components->get_num_elements();i++)
	{
		auto comp=m_components->get_element<Distribution>(i);
		comp->set_features(features);


	}

	auto dotdata=std::dynamic_pointer_cast<DotFeatures>(features);
	require(dotdata,"dynamic cast from CFeatures to CDotFeatures returned NULL");
	int32_t num_vectors=dotdata->get_num_vectors();

	// set data for EM
	auto em=std::make_shared<EMMixtureModel>();
	em->data.alpha=SGMatrix<float64_t>(num_vectors,m_components->get_num_elements());
	em->data.components=m_components;
	em->data.weights=m_weights;

	// run EM
	bool is_converged=em->iterate_em(m_max_iters,m_conv_tol);
	if (!is_converged)
		io::warn("max iterations reached. No convergence yet!");


	return true;
}

float64_t MixtureModel::get_log_model_parameter(int32_t num_param)
{
	require(num_param==1,"number of parameters in mixture model is 1"
	" (i.e. number of components). num_components should be 1. {} supplied",num_param);

	return std::log(static_cast<float64_t>(get_num_components()));
}

float64_t MixtureModel::get_log_derivative(int32_t num_param, int32_t num_example)
{
	not_implemented(SOURCE_LOCATION);
	return 0;
}

float64_t MixtureModel::get_log_likelihood_example(int32_t num_example)
{
	require(features,"features not set");
	require(features->get_feature_class() == C_DENSE,"Dense features required");
	require(features->get_feature_type() == F_DREAL,"Real features required");

	SGVector<float64_t> log_likelihood_component(m_components->get_num_elements());
	for (int32_t i=0;i<m_components->get_num_elements();i++)
	{
		auto ith_comp=m_components->get_element<Distribution>(i);
		log_likelihood_component[i] =
		    ith_comp->get_log_likelihood_example(num_example) +
		    std::log(m_weights[i]);


	}

	return Math::log_sum_exp(log_likelihood_component);
}

SGVector<float64_t> MixtureModel::get_weights() const
{
	return m_weights;
}

void MixtureModel::set_weights(SGVector<float64_t> weights)
{
	m_weights=weights;
}

std::shared_ptr<DynamicObjectArray> MixtureModel::get_components() const
{

	return m_components;
}

void MixtureModel::set_components(std::shared_ptr<DynamicObjectArray> components)
{
	m_components=components;
}

index_t MixtureModel::get_num_components() const
{
	return m_components->get_num_elements();
}

std::shared_ptr<Distribution> MixtureModel::get_component(index_t index) const
{
	require(index<get_num_components(),"index supplied ({}) is greater than total mixture components ({})"
																				,index,get_num_components());
	return m_components->get_element<Distribution>(index);
}

void MixtureModel::set_max_iters(int32_t max_iters)
{
	m_max_iters=max_iters;
}

int32_t MixtureModel::get_max_iters() const
{
	return m_max_iters;
}

void MixtureModel::set_convergence_tolerance(float64_t conv_tol)
{
	m_conv_tol=conv_tol;
}

float64_t MixtureModel::get_convergence_tolerance() const
{
	return m_conv_tol;
}

SGVector<float64_t> MixtureModel::sample()
{
	// TBD
	not_implemented(SOURCE_LOCATION);;
	return SGVector<float64_t>();
}

SGVector<float64_t> MixtureModel::cluster(SGVector<float64_t> point)
{
	// TBD
	not_implemented(SOURCE_LOCATION);;
	return point;
}

void MixtureModel::init()
{
	m_components=NULL;
	m_weights=SGVector<float64_t>();
	m_conv_tol=1e-8;
	m_max_iters=1000;

	SG_ADD((std::shared_ptr<SGObject>*)&m_components,"m_components","components of mixture");
	SG_ADD(&m_weights,"m_weights","weights of components");
	SG_ADD(&m_conv_tol,"m_conv_tol","convergence tolerance");
	SG_ADD(&m_max_iters,"m_max_iters","max number of iterations");
}
