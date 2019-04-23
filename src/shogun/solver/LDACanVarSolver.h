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

#ifndef LDA_CAN_VAR_SOLVER_H_
#define LDA_CAN_VAR_SOLVER_H_

#include <shogun/mathematics/linalg/LinalgEnums.h>
#include <shogun/solver/LDASolver.h>

namespace shogun
{

	template <typename T>
	class LDACanVarSolver : public LDASolver<T>
	{
	protected:
		/** Between covariance matrix */
		SGMatrix<T> m_between_cov;
		/** Number of dimensions in the projected space */
		index_t m_num_dim;
		/** Singular values threshold in svd */
		float64_t m_threshold;
		/** eigenvectors matrix */
		SGMatrix<T> m_eigenvectors;
		/** eigenvalues vector */
		SGVector<T> m_eigenvalues;
		/** use bdc-svd algorithm */
		bool m_bdc_svd;

		/**
		 * Compute between class covariance matrix.
		 */
		virtual void compute_between_cov();

		/**
		 * Compute the eigenvectors through the canonical variates algorithm.
		 */
		virtual void canvar();

	public:
		LDACanVarSolver(
		    std::shared_ptr<DenseFeatures<T>> features, std::shared_ptr<MulticlassLabels> labels,
		    index_t num_dim, float64_t gamma = 0.0, bool bdc_svd = true,
		    float64_t threshold = 0.01)
		    : LDASolver<T>(features, labels, gamma)
		{
			m_num_dim = num_dim;
			m_threshold = threshold;
			m_bdc_svd = bdc_svd;

			compute_between_cov();
			canvar();
		}

		/** @returns eigenvectors to project features into the transformed space
		 */
		SGMatrix<T> get_eigenvectors();

		/** @returns eigenvalues */
		SGVector<T> get_eigenvalues();
	};

	template <typename T>
	void LDACanVarSolver<T>::compute_between_cov()
	{
		index_t num_features = this->m_features->get_num_features();
		index_t num_class = this->m_labels->get_num_classes();

		m_between_cov = SGMatrix<T>(num_features, num_class);
		linalg::zero(m_between_cov);
		for (index_t i = 0; i < num_class; ++i)
		{
			auto col = m_between_cov.get_column(i);
			linalg::add(this->m_class_mean[i], this->m_mean, col, (T)1, (T)-1);
			linalg::scale(col, col, (T)sqrt(this->m_class_count[i]));
		}
	}

	template <typename T>
	void LDACanVarSolver<T>::canvar()
	{
		index_t num_features = this->m_features->get_num_features();
		index_t num_vectors = this->m_features->get_num_vectors();

		index_t r = Math::min(num_vectors, num_features);
		SGMatrix<T> U(num_features, r);
		SGVector<T> singularValues(r);

		// thin U SVD
		linalg::SVDAlgorithm svd_alg =
		    m_bdc_svd ? linalg::SVDAlgorithm::BidiagonalDivideConquer
		              : linalg::SVDAlgorithm::Jacobi;
		linalg::svd(
		    this->m_features->get_feature_matrix(), singularValues, U, true,
		    svd_alg);

		// basis to represent the solution
		SGMatrix<T> Q;
		// keep only the directions s.t. singular value > threshold
		if (num_features > num_vectors)
		{
			index_t j = 0;
			for (index_t i = 0; i < num_vectors; ++i)
				if (singularValues[i] > m_threshold)
					++j;
				else
					break;
			Q = SGMatrix<T>(U.matrix, num_features, j, false);
		}
		else
			Q = U;

		// modified between scatter (Sb)
		auto aux = linalg::matrix_prod(Q, m_between_cov, true, false);
		m_between_cov = linalg::matrix_prod(aux, aux, false, true);
		// modified within scatter (Sw)
		aux = linalg::matrix_prod(Q, this->m_within_cov, true, false);
		this->m_within_cov = linalg::matrix_prod(aux, Q);

		// To find svd(inverse(Sw)' * Sb * inverse(Sw))
		// solve Sb = (Sw' * X) * Sw
		// 1. get chol(Sw)
		// 2. solve chol(Sw)' * Y = Sb
		// 3. solve chol(Sw) * X = Y
		// 4. compute svd(X)
		auto chol = linalg::cholesky_factor(this->m_within_cov);
		SGMatrix<T> W(m_between_cov.num_rows, m_between_cov.num_cols);
		SGVector<T> eigenvalues(m_between_cov.num_rows);
		linalg::svd(
		    linalg::triangular_solver(
		        chol, linalg::transpose_matrix(
		                  linalg::triangular_solver(chol, m_between_cov))),
		    eigenvalues, W, true, svd_alg);

		m_eigenvectors = SGMatrix<T>(num_features, m_num_dim);
		linalg::zero(m_eigenvectors);
		m_eigenvalues = SGVector<T>(m_num_dim);

		auto Wt = linalg::matrix_prod(Q, W);
		for (index_t i = 0; i < m_num_dim; ++i)
		{
			m_eigenvectors.set_column(i, Wt.get_column(i));
			m_eigenvalues[i] = eigenvalues[i];
		}
	}

	template <typename T>
	SGMatrix<T> LDACanVarSolver<T>::get_eigenvectors()
	{
		return m_eigenvectors;
	}

	template <typename T>
	SGVector<T> LDACanVarSolver<T>::get_eigenvalues()
	{
		return m_eigenvalues;
	}
}

#endif // LDA_CAN_VAR_SOLVER_H_
