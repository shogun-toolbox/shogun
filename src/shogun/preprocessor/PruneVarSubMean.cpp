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

PruneVarSubMean::PruneVarSubMean(bool divide)
: DensePreprocessor<float64_t>()
{
	init();
	register_parameters();
	m_divide_by_std = divide;
}

PruneVarSubMean::~PruneVarSubMean()
{
	cleanup();
}

void PruneVarSubMean::fit(std::shared_ptr<Features> features)
{
	if (m_fitted)
		cleanup();

	auto simple_features = features->as<DenseFeatures<float64_t>>();
	auto num_examples = simple_features->get_num_vectors();
	auto num_features = simple_features->get_num_features();

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

	io::info("Reducing number of features from {} to {}", num_features, num_ok);

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

	m_fitted = true;
}

/// clean up allocated memory
void PruneVarSubMean::cleanup()
{
	m_idx=SGVector<int32_t>();
	m_mean=SGVector<float64_t>();
	m_std=SGVector<float64_t>();
	m_fitted = false;
}

SGMatrix<float64_t>
PruneVarSubMean::apply_to_matrix(SGMatrix<float64_t> matrix)
{
	assert_fitted();

	auto num_vectors = matrix.num_cols;
	auto result = matrix;
	result.num_rows = m_num_idx;

	for (auto i : range(num_vectors))
	{
		auto v_src = matrix.get_column(i);
		auto v_dst = result.get_column(i);

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
SGVector<float64_t> PruneVarSubMean::apply_to_feature_vector(SGVector<float64_t> vector)
{
	assert_fitted();

	SGVector<float64_t> out(m_num_idx);

	for (auto i : range(m_num_idx))
	{
		out[i] = vector[m_idx[i]] - m_mean[i];
		if (m_divide_by_std)
			out[i] /= m_std[i];
	}

	return out;
}

void PruneVarSubMean::init()
{
	m_fitted = false;
	m_divide_by_std = false;
	m_num_idx = 0;
	m_idx = SGVector<int32_t>();
	m_mean = SGVector<float64_t>();
	m_std = SGVector<float64_t>();
}

void PruneVarSubMean::register_parameters()
{
	SG_ADD(&m_divide_by_std, "divide_by_std", "Divide by standard deviation", ParameterProperties::HYPER);
	SG_ADD(&m_num_idx, "num_idx", "Number of elements in idx_vec");
	SG_ADD(&m_std, "std_vec", "Standard dev vector");
	SG_ADD(&m_mean, "mean_vec", "Mean vector");
	SG_ADD(&m_idx, "idx_vec", "Index vector");
}
