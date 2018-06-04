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
		{
			auto diff =
			    linalg::add(m_mean, feature_matrix.get_column(i), 1.0, -1.0);
			var[j] += linalg::dot(diff, diff);
		}
	}

	int32_t num_ok = 0;
	auto idx_ok = SGVector<int32_t>(num_features);

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

SGMatrix<float64_t>
CPruneVarSubMean::apply_to_matrix(SGMatrix<float64_t> matrix)
{
	ASSERT(m_initialized)

	int32_t num_vectors = matrix.num_cols;

	SGMatrix<float64_t> result(matrix.data(), m_num_idx, num_vectors);

	for (auto i : range(num_vectors))
	{
		auto v_src = matrix.get_column(i);
		auto v_dst = matrix.get_column(i);

		if (m_divide_by_std)
		{
			for (auto feat : range(m_num_idx))
				v_dst[feat]=(v_src[m_idx[feat]]-m_mean[feat])/m_std[feat];
		}
		else
		{
			for (auto feat : range(m_num_idx))
				v_dst[feat]=(v_src[m_idx[feat]]-m_mean[feat]);
		}
	}

	return result;
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
