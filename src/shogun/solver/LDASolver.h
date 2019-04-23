/*
 * Copyright (c) 2017, Shogun Toolbox Foundation
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:

 * 1. Redistributions of source code must retain the above copyright notice,
 * this list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 * this list of conditions and the following disclaimer in the documentation
 * and/or other materials provided with the distribution.
 *
 * 3. Neither the name of the copyright holder nor the names of its
 * contributors may be used to endorse or promote products derived from this
 * software without specific prior written permission.

 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
 * LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 * CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 * SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE.
 *
 */

#ifndef LDA_SOLVER_H_
#define LDA_SOLVER_H_

#include <shogun/base/SGObject.h>
#include <shogun/features/DenseFeatures.h>
#include <shogun/labels/MulticlassLabels.h>
#include <shogun/lib/config.h>
#include <shogun/mathematics/linalg/LinalgNamespace.h>
#include <vector>

namespace shogun
{

	template <typename T>
	class LDASolver
	{
	protected:
		std::shared_ptr<DenseFeatures<T>> m_features;
		std::shared_ptr<MulticlassLabels> m_labels;
		// Regularization parameter
		float64_t m_gamma;
		// Vector that holds the mean of each class
		std::vector<SGVector<T>> m_class_mean;
		// Vector of number of data points of each class
		std::vector<index_t> m_class_count;
		// Total mean vector
		SGVector<T> m_mean;
		// Within covariance matrix
		SGMatrix<T> m_within_cov;

		/**
		 * Compute the total mean and for each class the number of data points
		 * and its mean.
		 */
		virtual void compute_means();

		/**
		 * Compute within class covariance matrix.
		 */
		virtual void compute_within_cov();

	public:
		LDASolver(
		    std::shared_ptr<DenseFeatures<T>> features, std::shared_ptr<MulticlassLabels> labels,
		    float64_t gamma = 0.0)
		{

			m_features = features;
			m_labels = labels;
			m_gamma = gamma;

			compute_means();
			compute_within_cov();
		}

		~LDASolver()
		{
		}

		/** @return the vector of classes' mean */
		std::vector<SGVector<T>> get_class_mean();

		/** @return the number of data points of each class */
		std::vector<index_t> get_class_count();

		/** @return the total mean */
		SGVector<T> get_mean();

		/** @return the within covariance matrix */
		SGMatrix<T> get_within_cov();
	};

	template <typename T>
	void LDASolver<T>::compute_means()
	{
		index_t num_class = m_labels->get_num_classes();
		auto data = m_features->get_feature_matrix();

		m_class_mean = std::vector<SGVector<T>>(num_class);
		m_class_count = std::vector<index_t>(num_class);
		for (index_t i = 0; i < num_class; ++i)
		{
			m_class_mean[i] = SGVector<T>(data.num_rows);
			linalg::zero(m_class_mean[i]);
		}
		m_mean = SGVector<T>(data.num_rows);
		linalg::zero(m_mean);

		// calculate the total mean and the classes' mean.
		for (index_t i = 0; i < data.num_cols; ++i)
		{
			index_t c = (index_t)m_labels->get_label(i);
			++m_class_count[c];
			linalg::add_col_vec(data, i, m_class_mean[c], m_class_mean[c]);
		}
		for (index_t i = 0; i < num_class; ++i)
		{
			linalg::add(m_mean, m_class_mean[i], m_mean);
			linalg::scale(
			    m_class_mean[i], m_class_mean[i], 1 / (T)m_class_count[i]);
		}
		linalg::scale(m_mean, m_mean, 1 / (T)data.num_cols);
	}

	template <typename T>
	void LDASolver<T>::compute_within_cov()
	{
		index_t num_features = m_features->get_num_features();
		index_t num_vectors = m_features->get_num_vectors();
		index_t num_class = m_labels->get_num_classes();

		auto data = m_features->get_feature_matrix().clone();

		// Center data with respect to each data point's class
		for (index_t i = 0; i < data.num_cols; ++i)
			linalg::add_col_vec(
			    data, i, m_class_mean[m_labels->get_label(i)], data, (T)1.0,
			    (T)-1.0);

		// holds the feature matrix for each class
		std::vector<SGMatrix<T>> centered_class(num_class);
		std::vector<index_t> centered_class_col(num_class);

		m_within_cov = SGMatrix<T>(num_features, num_features);
		linalg::zero(m_within_cov);
		for (auto i = 0; i < num_class; ++i)
		{
			centered_class[i] = SGMatrix<T>(num_features, m_class_count[i]);
			linalg::zero(centered_class[i]);
		}
		for (index_t i = 0; i < num_vectors; ++i)
		{
			index_t c = (index_t)m_labels->get_label(i);
			centered_class[c].set_column(
			    centered_class_col[c], data.get_column(i));
			++centered_class_col[c];
		}
		for (index_t i = 0; i < num_class; ++i)
		{
			auto tmp = linalg::matrix_prod(
			    centered_class[i], centered_class[i], false, true);
			linalg::add(
			    m_within_cov, tmp, m_within_cov, (T)1.0,
			    ((T)m_class_count[i] / (m_class_count[i] - 1)));
		}

		if (m_gamma > 0.0)
		{
			T trace = linalg::trace(m_within_cov);
			SGMatrix<T> id(num_features, num_features);
			linalg::identity(id);
			linalg::add(
			    m_within_cov, id, m_within_cov, (T)(1.0 - m_gamma),
			    trace * ((T)m_gamma) / num_features);
		}
	}

	template <typename T>
	std::vector<SGVector<T>> LDASolver<T>::get_class_mean()
	{
		return m_class_mean;
	}

	template <typename T>
	std::vector<index_t> LDASolver<T>::get_class_count()
	{
		return m_class_count;
	}

	template <typename T>
	SGVector<T> LDASolver<T>::get_mean()
	{
		return m_mean;
	}

	template <typename T>
	SGMatrix<T> LDASolver<T>::get_within_cov()
	{
		return m_within_cov;
	}
}

#endif // LDA_SOLVER_H_
