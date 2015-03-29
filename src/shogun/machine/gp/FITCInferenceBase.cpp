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
#include <shogun/machine/gp/FITCInferenceBase.h>

#ifdef HAVE_EIGEN3

#include <shogun/machine/gp/GaussianLikelihood.h>
#include <shogun/mathematics/Math.h>
#include <shogun/labels/RegressionLabels.h>
#include <shogun/mathematics/eigen3.h>
#include <shogun/features/DenseFeatures.h>

using namespace shogun;
using namespace Eigen;

CFITCInferenceBase::CFITCInferenceBase() : CInferenceMethod()
{
	init();
}

void CFITCInferenceBase::check_features()
{
	REQUIRE(m_features, "Input features not set\n")
	REQUIRE(m_inducing_features, "Inducing features not set\n")
}

void CFITCInferenceBase::convert_features()
{
	CDotFeatures *feat_type=dynamic_cast<CDotFeatures *>(m_features);
	CDotFeatures *lat_type=dynamic_cast<CDotFeatures *>(m_inducing_features);
	REQUIRE(feat_type, "Input features (%s) must be DotFeatures"
		" or one of its subclasses\n", m_features->get_name())
	REQUIRE(lat_type, "Inducing features (%s) must be"
		" DotFeatures or one of its subclasses\n", m_inducing_features->get_name())
	REQUIRE(feat_type->get_dim_feature_space()==lat_type->get_dim_feature_space(),
		"The dim of feature spaces between"
		" input features (%d) and inducing features (%d) must be same\n",
		feat_type->get_dim_feature_space(),
		lat_type->get_dim_feature_space())
	if((m_features->get_feature_class()!=m_inducing_features->get_feature_class())||
		(m_features->get_feature_type()!=m_inducing_features->get_feature_type()))
	{
		if(m_features->get_feature_class()!=m_inducing_features->get_feature_class())
		{
			SG_WARNING("Input features (%s) and inducing features (%s) are"
				" difference classes\n", m_features->get_name(),
				m_inducing_features->get_name());
		}
		if(m_features->get_feature_type()!=m_inducing_features->get_feature_type())
		{
			SG_WARNING("Input features and inducing features are difference types\n");
		}
		SG_WARNING("Input features and inducing features may be deleted\n");
		SGMatrix<float64_t> feat_m=feat_type->get_computed_dot_feature_matrix();
		SG_UNREF(m_features);
		m_features=new CDenseFeatures<float64_t>(feat_m);
		SG_REF(m_features);
		SGMatrix<float64_t> lat_m=lat_type->get_computed_dot_feature_matrix();
		SG_UNREF(m_inducing_features)
		m_inducing_features=new CDenseFeatures<float64_t>(lat_m);
		SG_REF(m_inducing_features);
	}
}

CFITCInferenceBase::CFITCInferenceBase(CKernel* kern, CFeatures* feat,
		CMeanFunction* m, CLabels* lab, CLikelihoodModel* mod, CFeatures* lat)
		: CInferenceMethod(kern, feat, m, lab, mod)
{
	init();
	set_inducing_features(lat);
}

void CFITCInferenceBase::init()
{
	SG_ADD((CSGObject**)&m_inducing_features, "inducing_features", "inducing features",
			MS_AVAILABLE, GRADIENT_AVAILABLE);
	SG_ADD(&m_ind_noise, "inducing_noise", "noise about inducing potins",
		MS_AVAILABLE, GRADIENT_AVAILABLE);
	SG_ADD(&m_mu, "mu", "mean vector of the approximation to the posterior", MS_NOT_AVAILABLE);
	SG_ADD(&m_Sigma, "Sigma", "covariance matrix of the approximation to the posterior", MS_NOT_AVAILABLE);
	SG_ADD(&m_ktrtr_diag, "ktrtr_diag", "diagonal elements of kernel matrix m_ktrtr", MS_NOT_AVAILABLE);

	m_inducing_features=NULL;
	m_ind_noise=1e-10;
}

void CFITCInferenceBase::set_inducing_noise(float64_t noise)
{
	REQUIRE(!(noise<0), "Noise for inducing points must be non-negative");
	m_ind_noise=noise;
}

float64_t CFITCInferenceBase::get_inducing_noise()
{
	return m_ind_noise;
}

CFITCInferenceBase::~CFITCInferenceBase()
{
	SG_UNREF(m_inducing_features);
}

void CFITCInferenceBase::check_members() const
{
	CInferenceMethod::check_members();

	REQUIRE(m_inducing_features, "Inducing features should not be NULL\n")
	REQUIRE(m_inducing_features->get_num_vectors(),
			"Number of inducing features must be greater than zero\n")
}

SGVector<float64_t> CFITCInferenceBase::get_alpha()
{
	if (parameter_hash_changed())
		update();

	SGVector<float64_t> result(m_alpha);
	return result;
}

SGMatrix<float64_t> CFITCInferenceBase::get_cholesky()
{
	if (parameter_hash_changed())
		update();

	SGMatrix<float64_t> result(m_L);
	return result;
}

void CFITCInferenceBase::update_train_kernel()
{
	//time complexity can be O(m*n) if the TO DO is done
	check_features();
	convert_features();

	m_kernel->init(m_features, m_features);
	m_ktrtr_diag=m_kernel->get_kernel_diagonal();

	// create kernel matrix for inducing features
	m_kernel->init(m_inducing_features, m_inducing_features);
	m_kuu=m_kernel->get_kernel_matrix();

	// create kernel matrix for inducing and training features
	m_kernel->init(m_inducing_features, m_features);
	m_ktru=m_kernel->get_kernel_matrix();
}

#endif /* HAVE_EIGEN3 */
