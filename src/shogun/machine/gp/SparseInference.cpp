/*
 * Copyright (c) The Shogun Machine Learning Toolbox
 * Written (W) 2015 Wu Lin
 * Written (W) 2013 Roman Votyakov
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
#include <shogun/machine/gp/SparseInference.h>


#include <shogun/machine/gp/GaussianLikelihood.h>
#include <shogun/mathematics/Math.h>
#include <shogun/labels/RegressionLabels.h>
#include <shogun/mathematics/eigen3.h>

#include <utility>

using namespace shogun;
using namespace Eigen;

SparseInference::SparseInference() : Inference()
{
	init();
}

void SparseInference::check_features()
{
	require(m_features, "Input features not set");
}

void SparseInference::convert_features()
{
	auto feat_type=m_features->as<DotFeatures>();

	SGMatrix<float64_t>lat_m(m_inducing_features.matrix,
		m_inducing_features.num_rows,m_inducing_features.num_cols,false);
	auto lat_type=std::make_shared<DenseFeatures<float64_t>>(lat_m);

	require(feat_type, "Input features ({}) must be DotFeatures"
		" or one of its subclasses", m_features->get_name());
	require(feat_type->get_dim_feature_space()==lat_type->get_dim_feature_space(),
		"The dim of feature spaces between"
		" input features ({}) and inducing features ({}) must be same",
		feat_type->get_dim_feature_space(),
		lat_type->get_dim_feature_space());
	if((m_features->get_feature_class()!=lat_type->get_feature_class())||
		(m_features->get_feature_type()!=lat_type->get_feature_type()))
	{
		if(m_features->get_feature_class()!=lat_type->get_feature_class())
		{
			io::warn("Input features ({}) and inducing features ({}) are"
				" difference classes", m_features->get_name(),
				lat_type->get_name());
		}
		if(m_features->get_feature_type()!=lat_type->get_feature_type())
		{
			io::warn("Input features and inducing features are difference types");
		}
		io::warn("Input features may be deleted");
		SGMatrix<float64_t> feat_m=feat_type->get_computed_dot_feature_matrix();

		m_features=std::make_shared<DenseFeatures<float64_t>>(feat_m);

	}

}

SparseInference::SparseInference(std::shared_ptr<Kernel> kern, std::shared_ptr<Features> feat,
		std::shared_ptr<MeanFunction> m, std::shared_ptr<Labels> lab, std::shared_ptr<LikelihoodModel> mod, std::shared_ptr<Features> lat)
		: Inference(std::move(kern), std::move(feat), std::move(m), std::move(lab), std::move(mod))
{
	init();
	set_inducing_features(std::move(lat));
}

void SparseInference::init()
{
	SG_ADD(&m_inducing_features, "inducing_features", "inducing features",
			ParameterProperties::HYPER | ParameterProperties::GRADIENT);
	SG_ADD(&m_log_ind_noise, "log_inducing_noise", "noise about inducing potins in log domain",
		ParameterProperties::HYPER | ParameterProperties::GRADIENT);
	SG_ADD(&m_mu, "mu", "mean vector of the approximation to the posterior");
	SG_ADD(&m_Sigma, "Sigma", "covariance matrix of the approximation to the posterior");
	SG_ADD(&m_ktrtr_diag, "ktrtr_diag", "diagonal elements of kernel matrix m_ktrtr");

	m_log_ind_noise = std::log(1e-10);
	m_inducing_features=SGMatrix<float64_t>();
}

void SparseInference::set_inducing_noise(float64_t noise)
{
	require(noise>0, "Noise ({}) for inducing points must be postive",noise);
	m_log_ind_noise = std::log(noise);
}

float64_t SparseInference::get_inducing_noise()
{
	return std::exp(m_log_ind_noise);
}

SparseInference::~SparseInference()
{
}

void SparseInference::check_members() const
{
	Inference::check_members();

	require(m_inducing_features.num_rows, "Inducing features should not be empty");
	require(m_inducing_features.num_cols, "Inducing features should not be empty");
}

SGVector<float64_t> SparseInference::get_alpha()
{
	if (parameter_hash_changed())
		update();

	SGVector<float64_t> result(m_alpha);
	return result;
}

SGMatrix<float64_t> SparseInference::get_cholesky()
{
	if (parameter_hash_changed())
		update();

	SGMatrix<float64_t> result(m_L);
	return result;
}

void SparseInference::update_train_kernel()
{
	//time complexity can be O(m*n) if the TO DO is done
	check_features();
	convert_features();

	m_kernel->init(m_features, m_features);
	m_ktrtr_diag=m_kernel->get_kernel_diagonal();

	auto inducing_features=get_inducing_features();

	// create kernel matrix for inducing features
	m_kernel->init(inducing_features, inducing_features);
	m_kuu=m_kernel->get_kernel_matrix();

	// create kernel matrix for inducing and training features
	m_kernel->init(inducing_features, m_features);
	m_ktru=m_kernel->get_kernel_matrix();


}

