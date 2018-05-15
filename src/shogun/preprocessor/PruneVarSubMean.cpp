/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Soeren Sonnenburg, Viktor Gal, Evgeniy Andreev, Evan Shelhamer, 
 *          Sergey Lisitsyn, Bjoern Esser
 */

#include <shogun/features/Features.h>
#include <shogun/io/SGIO.h>
#include <shogun/mathematics/Math.h>
#include <shogun/mathematics/linalg/LinalgNamespace.h>
#include <shogun/preprocessor/DensePreprocessor.h>
#include <shogun/preprocessor/PruneVarSubMean.h>

using namespace shogun;

CPruneVarSubMean::CPruneVarSubMean(bool divide)
: CDensePreprocessor<float64_t>()
{
	init();
	register_parameters();
	m_divide_by_std = divide;
}

CPruneVarSubMean::~CPruneVarSubMean()
{
	cleanup();
}

void CPruneVarSubMean::fit(CFeatures* features)
{
	if (m_initialized)
		cleanup();

	auto simple_features = features->as<CDenseFeatures<float64_t>>();
	int32_t num_examples = simple_features->get_num_vectors();
	int32_t num_features = simple_features->get_num_features();

	m_idx = SGVector<int32_t>();
	m_std = SGVector<float64_t>();

	SGVector<float64_t> var(num_features);

	auto feature_matrix = simple_features->get_feature_matrix();

	// compute mean
	m_mean = linalg::rowwise_sum(feature_matrix);
	linalg::scale(m_mean, m_mean, 1.0 / num_examples);

	// compute var
	for (auto i : range(num_examples))
	{
		for (auto j : range(num_features))
			var[j] += CMath::sq(
			    m_mean[j] - feature_matrix.matrix[i * num_features + j]);
	}

	int32_t num_ok = 0;
	int32_t* idx_ok = SG_MALLOC(int32_t, num_features);

	for (auto j : range(num_features))
	{
		var[j] /= num_examples;

		if (var[j] >= 1e-14)
		{
			idx_ok[num_ok] = j;
			num_ok++;
		}
	}

	SG_INFO("Reducing number of features from %i to %i\n", num_features, num_ok)

	m_idx.resize_vector(num_ok);
	SGVector<float64_t> new_mean(num_ok);
	m_std.resize_vector(num_ok);

	for (auto j : range(num_ok))
	{
		m_idx[j] = idx_ok[j];
		new_mean[j] = m_mean[idx_ok[j]];
		m_std[j] = std::sqrt(var[idx_ok[j]]);
	}
	m_num_idx = num_ok;
	SG_FREE(idx_ok);
	m_mean = new_mean;

	m_initialized = true;
}

/// clean up allocated memory
void CPruneVarSubMean::cleanup()
{
	m_idx=SGVector<int32_t>();
	m_mean=SGVector<float64_t>();
	m_std=SGVector<float64_t>();
	m_initialized = false;
}

/// apply preproc on feature matrix
/// result in feature matrix
/// return pointer to feature_matrix, i.e. f->get_feature_matrix();
SGMatrix<float64_t> CPruneVarSubMean::apply_to_feature_matrix(CFeatures* features)
{
	ASSERT(m_initialized)

	int32_t num_vectors=0;
	int32_t num_features=0;

	auto simple_features = features->as<CDenseFeatures<float64_t>>();
	auto m = simple_features->get_feature_matrix();

	SG_INFO("get Feature matrix: %ix%i\n", num_vectors, num_features)
	SG_INFO("Preprocessing feature matrix\n")
	for (int32_t vec=0; vec<num_vectors; vec++)
	{
		float64_t* v_src=&m[num_features*vec];
		float64_t* v_dst=&m[m_num_idx*vec];

		if (m_divide_by_std)
		{
			for (int32_t feat=0; feat<m_num_idx; feat++)
				v_dst[feat]=(v_src[m_idx[feat]]-m_mean[feat])/m_std[feat];
		}
		else
		{
			for (int32_t feat=0; feat<m_num_idx; feat++)
				v_dst[feat]=(v_src[m_idx[feat]]-m_mean[feat]);
		}
	}

	simple_features->set_num_features(m_num_idx);
	simple_features->get_feature_matrix(num_features, num_vectors);
	SG_INFO("new Feature matrix: %ix%i\n", num_vectors, num_features)

	return simple_features->get_feature_matrix();
}

/// apply preproc on single feature vector
/// result in feature matrix
SGVector<float64_t> CPruneVarSubMean::apply_to_feature_vector(SGVector<float64_t> vector)
{
	float64_t* ret=NULL;

	if (m_initialized)
	{
		ret=SG_MALLOC(float64_t, m_num_idx);

		if (m_divide_by_std)
		{
			for (int32_t i=0; i<m_num_idx; i++)
				ret[i]=(vector.vector[m_idx[i]]-m_mean[i])/m_std[i];
		}
		else
		{
			for (int32_t i=0; i<m_num_idx; i++)
				ret[i]=(vector.vector[m_idx[i]]-m_mean[i]);
		}
	}
	else
	{
		ret=SG_MALLOC(float64_t, vector.vlen);
		for (int32_t i=0; i<vector.vlen; i++)
			ret[i]=vector.vector[i];
	}

	return SGVector<float64_t>(ret,m_num_idx);
}

void CPruneVarSubMean::init()
{
	m_initialized = false;
	m_divide_by_std = false;
	m_num_idx = 0;
	m_idx = SGVector<int32_t>();
	m_mean = SGVector<float64_t>();
	m_std = SGVector<float64_t>();
}

void CPruneVarSubMean::register_parameters()
{
	SG_ADD(&m_initialized, "initialized", "The prerpocessor is initialized",  MS_NOT_AVAILABLE);
	SG_ADD(&m_divide_by_std, "divide_by_std", "Divide by standard deviation", MS_AVAILABLE);
	SG_ADD(&m_num_idx, "num_idx", "Number of elements in idx_vec", MS_NOT_AVAILABLE);
	SG_ADD(&m_std, "std_vec", "Standard dev vector", MS_NOT_AVAILABLE);
	SG_ADD(&m_mean, "mean_vec", "Mean vector", MS_NOT_AVAILABLE);
	SG_ADD(&m_idx, "idx_vec", "Index vector", MS_NOT_AVAILABLE);
}
