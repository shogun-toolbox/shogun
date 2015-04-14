/*
 * This software is distributed under BSD Clause 3 license (see LICENSE file).
 *
 * Written (W) 2015 Esben Soerig
 */

#include <shogun/kernel/SpectralMixtureKernel.h>

using namespace shogun;

CSpectralMixtureKernel::CSpectralMixtureKernel() : CDotKernel()
{
	/* We can't really say anything about number of components at this point
	   We have to leave the object in an unusable state until a set of feature
	   has been received
	   */
	init();
}

CSpectralMixtureKernel::CSpectralMixtureKernel(CDotFeatures* l, CDotFeatures* r,
	int32_t num_components, int32_t size) : CDotKernel(size)
{
	init();
	CDotKernel::init(l, r);
	init_components(num_components);
}

CSpectralMixtureKernel::CSpectralMixtureKernel(const SGVector<float64_t>& weights, 
											   const SGVector<float64_t>& means, 
											   const SGVector<float64_t>& stddeviations, 
											   int32_t size) : CDotKernel(size)
{
	init();
	set_component_parameters(weights, means, stddeviations);
}

void CSpectralMixtureKernel::init()
{
	SG_ADD(&m_weights, "weights",
		"Vector of weights for each mixture component", MS_AVAILABLE, GRADIENT_AVAILABLE);
	SG_ADD(&m_means, "means",
		"Vector of means for each mixture component", MS_AVAILABLE, GRADIENT_AVAILABLE);
	SG_ADD(&m_stddeviations, "stddeviations",
		"Vector of standard deviations for each mixture component", MS_AVAILABLE, GRADIENT_AVAILABLE);
}

bool CSpectralMixtureKernel::init_components(int32_t num_components)
{
	REQUIRE(has_features(), "Features not set - cannot initialize components.");

	m_weights = SGVector<float64_t>(num_components);
	m_weights.set_const(1.0/num_components);

	m_means = SGVector<float64_t>(num_components);
	m_means.set_const(0.0);

	m_stddeviations = SGVector<float64_t>(num_components);
	m_stddeviations.set_const(1.0);

	return true;
}

int32_t CSpectralMixtureKernel::get_num_components() const
{
	return m_weights.size();
}

void CSpectralMixtureKernel::set_component_parameters(const SGVector<float64_t>& weights, 
													  const SGVector<float64_t>& means,
													  const SGVector<float64_t>& stddeviations)
{
	bool same_size = weights.size()==means.size() && weights.size()==stddeviations.size();
	REQUIRE(same_size, "Weights, means, and stddeviations vectors must be of equal size");

	m_weights = weights;
	m_means = means;
	m_stddeviations = stddeviations;
}

float64_t CSpectralMixtureKernel::compute(int32_t idx_a, int32_t idx_b)
{
	REQUIRE(has_features(), "Features not set - cannot compute kernel value.");
	REQUIRE(m_weights.size() > 0, "Mixture components not set - cannot compute kernel values");

	CDotFeatures* dotlhs = dynamic_cast<CDotFeatures*>(lhs);
	CDotFeatures* dotrhs = dynamic_cast<CDotFeatures*>(rhs);

	REQUIRE(dotlhs != NULL, "Left-hand side features could not be cast to CDotFeautres");
	REQUIRE(dotrhs != NULL, "Right-hand side features could not be cast to CDotFeatures");

	float64_t result = 0.0;
	float64_t distance_sq = dotlhs->dot(idx_a, dotlhs, idx_a) + dotrhs->dot(idx_b, dotrhs, idx_b) - 2*dotlhs->dot(idx_a, dotrhs, idx_b);

	for(int i=0; i<m_weights.size(); i++)
		result += CMath::pow(m_weights[i],2)*CMath::exp(-2*CMath::pow(CMath::PI,2)*distance_sq*CMath::pow(m_stddeviations[i],2))*CMath::cos(2*CMath::PI*CMath::sqrt(distance_sq)*m_means[i]);

	return result;
}

SGVector<float64_t> CSpectralMixtureKernel::get_squared_features(CDotFeatures* features)
{
	SGVector<float64_t> sqr_features = SGVector<float64_t>(features->get_num_vectors());
	
	for (int i=0; i<sqr_features.size(); i++)
		sqr_features[i] = features->dot(i, features, i);

	return sqr_features;
}

SGMatrix<float64_t> CSpectralMixtureKernel::get_square_distance_matrix()
{
	CDotFeatures* dot_lhs = dynamic_cast<CDotFeatures*>(lhs);
	CDotFeatures* dot_rhs = dynamic_cast<CDotFeatures*>(rhs);

	REQUIRE(dot_lhs != NULL, "Left-hand-side features are not of type CDotFeatures");
	REQUIRE(dot_rhs != NULL, "Right-hand-side features are not of type CDotFeatures");
	
	SGMatrix<float64_t> lhs_mtrx = dot_lhs->get_computed_dot_feature_matrix();
	SGMatrix<float64_t> rhs_mtrx = dot_rhs->get_computed_dot_feature_matrix();

	/* rhs^T . lhs: index (i, j) is dot product of lhs_i and rhs_j.
	   result scaled by minus two for distance computation */
	SGMatrix<float64_t> dot_mtrx = SGMatrix<float64_t>::matrix_multiply(rhs_mtrx, lhs_mtrx, true, false, -2.0);

	SGVector<float64_t> sqr_lhs = get_squared_features(dot_lhs);
	SGVector<float64_t> sqr_rhs = get_squared_features(dot_rhs);

	for (int i=0; i<dot_mtrx.num_cols; i++)
	{
		/* To each column vector is added a scalar: lhs_i^2
		   and the vector [rhs_1^2 ... rhs_n^2]^T*/
		float64_t* col_vector_ptr = dot_mtrx.get_column_vector(i);

		SGVector<float64_t>::add_scalar(sqr_lhs[i], col_vector_ptr, dot_mtrx.num_rows);
		SGVector<float64_t>::add(col_vector_ptr, 1.0, col_vector_ptr, 1.0, sqr_rhs, dot_mtrx.num_rows);
	}
	
	return dot_mtrx;
}

void CSpectralMixtureKernel::compute_weight_derivative(SGMatrix<float64_t>& squared_distances, index_t index)
{
	for (int i=0; i<squared_distances.num_rows; i++)
	{
		for (int j=0; j<squared_distances.num_cols; j++)
		{
			// 2*w_i*exp(-2*pi^2*dist^2*sigma_i^2) * cos(2*pi*dist*mu_i)
			float64_t sqr_dist = squared_distances(i,j);
			float64_t exp_factor = CMath::exp(-2*CMath::pow(CMath::PI,2)*sqr_dist*CMath::pow(m_stddeviations[index],2));
			float64_t cos_factor = CMath::cos(2*CMath::PI*CMath::sqrt(sqr_dist)*m_means[index]);
			squared_distances(i, j) = 2*m_weights[index]*exp_factor*cos_factor;
		}
	}
}

void CSpectralMixtureKernel::compute_mean_derivative(SGMatrix<float64_t>& squared_distances, index_t index)
{
	for (int i=0; i<squared_distances.num_rows; i++)
	{
		for (int j=0; j<squared_distances.num_cols; j++)
		{
			// -2*pi*dist*mu_i^2*exp(-2*pi^2*dist^2*sigma_i^2)*sin(2*pi*dist*mu_i)
			float64_t sqr_dist = squared_distances(i,j);
			float64_t outer_factor = -2*CMath::PI*CMath::sqrt(sqr_dist)*CMath::pow(m_weights[index],2);
			float64_t exp_factor = CMath::exp(-2*CMath::pow(CMath::PI, 2)*sqr_dist*CMath::pow(m_stddeviations[index], 2));
			float64_t sin_factor = CMath::sin(2*CMath::PI*CMath::sqrt(sqr_dist)*m_means[index]);
			
			squared_distances(i, j) = outer_factor * exp_factor * sin_factor;
		}
	}
}

void CSpectralMixtureKernel::compute_stddeviation_derivative(SGMatrix<float64_t>& squared_distances, index_t index)
{
	for (int i=0; i<squared_distances.num_rows; i++)
	{
		for (int j=0; j<squared_distances.num_cols; j++)
		{
			//-4*pi^2*dist^2*sigma_i*mu_i^2 * exp(-2*pi^2*dist^2*sigma_i^2) * cos(2*pi*dist*mu_i)
			float64_t sqr_dist = squared_distances(i,j);
			float64_t pi_sqr = CMath::pow(CMath::PI, 2);
			float64_t outer_factor = -4*pi_sqr*sqr_dist*m_stddeviations[index]*CMath::pow(m_weights[index], 2);
			float64_t exp_factor = CMath::exp(-2*pi_sqr*sqr_dist*CMath::pow(m_stddeviations[index], 2));
			float64_t cos_factor = CMath::cos(2*CMath::PI*CMath::sqrt(sqr_dist)*m_means[index]);
			
			squared_distances(i, j) = outer_factor * exp_factor * cos_factor;
		}
	}
}

SGMatrix<float64_t> CSpectralMixtureKernel::get_parameter_gradient(
				const TParameter* param, index_t index)
{
	REQUIRE(lhs, "Left-hand-side features not set!\n");
	REQUIRE(rhs, "Right-hand-side features not set!\n");

	if (!strcmp(param->m_name, "weights")) 
	{
		SGMatrix<float64_t> sqr_dists = get_square_distance_matrix();
		compute_weight_derivative(sqr_dists, index);

		return sqr_dists;
	}
	else if (!strcmp(param->m_name, "means"))
	{
		SGMatrix<float64_t> sqr_dists = get_square_distance_matrix();
		compute_mean_derivative(sqr_dists, index);

		return sqr_dists;
	}
	else if (!strcmp(param->m_name, "stddeviations"))
	{
		SGMatrix<float64_t> sqr_dists = get_square_distance_matrix();
		compute_stddeviation_derivative(sqr_dists, index);

		return sqr_dists;
	}
	else
	{
		SG_ERROR("Can't compute derivative wrt %s parameter\n", param->m_name);
		return SGMatrix<float64_t>();
	}
}
