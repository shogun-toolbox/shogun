/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Bjoern Esser, Evangelos Anagnostopoulos
 */

#include <shogun/features/DenseFeatures.h>
#include <shogun/features/RandomKitchenSinksDotFeatures.h>
#include <typeinfo>
#include <utility>

namespace shogun
{

class CRKSFunctions;

RandomKitchenSinksDotFeatures::RandomKitchenSinksDotFeatures()
	: RandomMixin<DotFeatures>()
{
	init(NULL, 0);
}

RandomKitchenSinksDotFeatures::RandomKitchenSinksDotFeatures(
	std::shared_ptr<DotFeatures> dataset, int32_t K)
{
	init(std::move(dataset), K);
}

RandomKitchenSinksDotFeatures::RandomKitchenSinksDotFeatures(
	std::shared_ptr<DotFeatures> dataset, int32_t K, SGMatrix<float64_t> coeff)
{
	init(std::move(dataset), K);
	random_coeff = coeff;
}

SGMatrix<float64_t> RandomKitchenSinksDotFeatures::generate_random_coefficients()
{
	SGVector<float64_t> vec = generate_random_parameter_vector();
	SGMatrix<float64_t> random_params(vec.vlen, num_samples);
	for (index_t dim=0; dim<random_params.num_rows; dim++)
		random_params(dim, 0) = vec[dim];

	for (index_t sample=1; sample<num_samples; sample++)
	{
		vec = generate_random_parameter_vector();
		for (index_t dim=0; dim<random_params.num_rows; dim++)
			random_params(dim, sample) = vec[dim];
	}
	return random_params;
}

RandomKitchenSinksDotFeatures::RandomKitchenSinksDotFeatures(const std::shared_ptr<File>& loader)
{
	not_implemented(SOURCE_LOCATION);;
}

RandomKitchenSinksDotFeatures::RandomKitchenSinksDotFeatures(
	const RandomKitchenSinksDotFeatures& orig)
{
	init(orig.feats, orig.num_samples);
	random_coeff = orig.random_coeff;
}

RandomKitchenSinksDotFeatures::~RandomKitchenSinksDotFeatures()
{

}

void RandomKitchenSinksDotFeatures::init(std::shared_ptr<DotFeatures> dataset,
	int32_t K)
{
	feats = std::move(dataset);


	num_samples = K;

	SG_ADD((std::shared_ptr<SGObject>* ) &feats, "feats", "Features to work on");
	SG_ADD(
		&random_coeff, "random_coeff", "Random function parameters");
}

int32_t RandomKitchenSinksDotFeatures::get_dim_feature_space() const
{
	return num_samples;
}

float64_t RandomKitchenSinksDotFeatures::dot(int32_t vec_idx1, std::shared_ptr<DotFeatures> df,
	int32_t vec_idx2) const
{
	ASSERT(typeid(*this) == typeid(*df));
	auto other = std::static_pointer_cast<RandomKitchenSinksDotFeatures>(df);
	ASSERT(get_dim_feature_space()==other->get_dim_feature_space());

	float64_t dot_product = 0;
	for (index_t i=0; i<num_samples; i++)
	{
		float64_t tmp_dot_1 = dot(vec_idx1, i);
		float64_t tmp_dot_2 = other->dot(vec_idx2, i);

		tmp_dot_1 = post_dot(tmp_dot_1, i);
		tmp_dot_2 = other->post_dot(tmp_dot_2, i);
		dot_product += tmp_dot_1 * tmp_dot_2;
	}
	return dot_product;
}

float64_t RandomKitchenSinksDotFeatures::dot(
	int32_t vec_idx1, const SGVector<float64_t>& vec2) const
{
	SG_TRACE("entering dense_dot()");
	ASSERT(vec2.size() == get_dim_feature_space());

	float64_t dot_product = 0;
	for (index_t i=0; i<num_samples; i++)
	{
		float64_t tmp_dot = dot(vec_idx1, i);
		tmp_dot = post_dot(tmp_dot, i);
		dot_product += tmp_dot * vec2[i];
	}
	SG_TRACE("Leaving dense_dot()");
	return dot_product;
}

void RandomKitchenSinksDotFeatures::add_to_dense_vec(float64_t alpha,
	int32_t vec_idx1, float64_t* vec2, int32_t vec2_len, bool abs_val) const
{
	SG_TRACE("Entering add_to_dense()");
	ASSERT(vec2_len == get_dim_feature_space());

	for (index_t i=0; i<num_samples; i++)
	{
		float64_t tmp_dot = dot(vec_idx1, i);
		tmp_dot = post_dot(tmp_dot, i);
		if (abs_val)
			vec2[i] += Math::abs(alpha * tmp_dot);
		else
			vec2[i] += alpha * tmp_dot;
	}
	SG_TRACE("Leaving add_to_dense()");
}

int32_t RandomKitchenSinksDotFeatures::get_nnz_features_for_vector(int32_t num) const
{
	return num_samples;
}

void* RandomKitchenSinksDotFeatures::get_feature_iterator(int32_t vector_index)
{
	not_implemented(SOURCE_LOCATION);;
	return NULL;
}

bool RandomKitchenSinksDotFeatures::get_next_feature(int32_t& index,
	float64_t& value, void* iterator)
{
	not_implemented(SOURCE_LOCATION);;
	return false;
}

void RandomKitchenSinksDotFeatures::free_feature_iterator(void* iterator)
{
	not_implemented(SOURCE_LOCATION);;
}

EFeatureType RandomKitchenSinksDotFeatures::get_feature_type() const
{
	return F_DREAL;
}

EFeatureClass RandomKitchenSinksDotFeatures::get_feature_class() const
{
	return C_DENSE;
}

int32_t RandomKitchenSinksDotFeatures::get_num_vectors() const
{
	return feats->get_num_vectors();
}

const char* RandomKitchenSinksDotFeatures::get_name() const
{
	return "RandomKitchenSinksDotFeatures";
}

std::shared_ptr<Features> RandomKitchenSinksDotFeatures::duplicate() const
{
	not_implemented(SOURCE_LOCATION);;
	return NULL;
}

SGMatrix<float64_t> RandomKitchenSinksDotFeatures::get_random_coefficients()
{
	return random_coeff;
}

float64_t RandomKitchenSinksDotFeatures::dot(index_t vec_idx, index_t par_idx) const
{
	auto vec2 = random_coeff.get_column(par_idx).slice(0, feats->get_dim_feature_space());
	return feats->dot(vec_idx, vec2);
}

float64_t RandomKitchenSinksDotFeatures::post_dot(float64_t dot_result, index_t par_idx) const
{
	return dot_result;
}

}
